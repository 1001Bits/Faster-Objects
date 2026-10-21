#include "Config.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace Config {
    namespace {
        namespace fs = std::filesystem;
        using json = nlohmann::json;

        constexpr auto kPath = "Data/SKSE/Plugins/SkyrimPrecombines.ini";

        struct Name { std::string_view ini /*lower case*/, key, display, section; };
        // The shipped file's settings (lower case) and the internal names their values are read under; the spelling and
        // section a menu change is written under when the file has no line for it.
        constexpr std::array kNames{
            Name{"benabled","enabled","bEnabled","General"},
            Name{"bruntimebake","runtime_bake","bRuntimeBake","General"},
            Name{"scachefolder","cache_folder","sCacheFolder","General"},
            Name{"bhiz","hiz_occlusion","bHiZ","Occlusion"},
            Name{"fhizminframems","hiz_min_frame_ms","fHiZMinFrameMs","Occlusion"},
            Name{"fhizminyield","hiz_min_yield","fHiZMinYield","Occlusion"},
            Name{"bdepthculling","vr_depth_culling","bDepthCulling","VR"},
            Name{"fminoccludeeextent","vr_min_occludee_extent","fMinOccludeeExtent","VR"},
            Name{"bdevelopershortcuts","developer_shortcuts","bDeveloperShortcuts","Developer"},
        };

        std::string_view Trim(std::string_view t) {
            while(!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.remove_prefix(1);
            while(!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.remove_suffix(1);
            return t;
        }
        std::string Lower(std::string_view t) {
            std::string out(t);
            std::transform(out.begin(),out.end(),out.begin(),[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return out;
        }
        std::string_view Unquote(std::string_view t) { return t.size()>=2 && t.front()=='"' && t.back()=='"' ? t.substr(1,t.size()-2) : t; }
        // A b, f or i value without a trailing comment or quotes, as the game's own INIs allow (bHiZ=0 ; testing, bHiZ="0");
        // neither ';' nor '#' is part of such a value. An s value is kept as written: a folder may contain both.
        std::string_view Plain(std::string_view text) { return Trim(Unquote(Trim(text.substr(0,text.find_first_of(";#"))))); }

        // A shipped setting's value by its prefix; nothing when it does not read as that type.
        std::optional<json> Typed(char prefix,std::string_view text) {
            const auto value=Lower(text);
            try {
                switch(prefix) {
                case 'b':
                    if(value=="1" || value=="true") return json(true);
                    if(value=="0" || value=="false") return json(false);
                    return std::nullopt;
                case 'f': case 'i': {
                    // from_chars: the decimal point whatever the process locale.
                    double number=0.0;
                    const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),number);
                    if(error!=std::errc() || end!=value.data()+value.size() || !std::isfinite(number)) return std::nullopt;
                    return json(number);
                }
                case 's': return json(std::string(Unquote(text)));
                }
            } catch(const std::exception&) {}
            return std::nullopt;
        }

        json Read() {
            json config=json::object();
            if(fs::exists("Data/SKSE/Plugins/SkyrimPrecombines.json"))
                logger::warn("SkyrimPrecombines.json (an older version's settings) is not read; the settings are in SkyrimPrecombines.ini");
            std::ifstream in(kPath);
            // A warning: a DLL-only install or a deleted file is the likeliest reason a setting seems ignored.
            if(!in) { logger::warn("No SkyrimPrecombines.ini; default settings"); return config; }
            std::string line;
            size_t number=0,values=0;
            while(std::getline(in,line)) {
                ++number;
                // UTF-16 (Notepad's "Unicode", PowerShell 5's > and Out-File) has a NUL beside every character: one error and
                // the defaults, not a warning for each line.
                if(number==1 && (line.starts_with("\xFF\xFE") || line.starts_with("\xFE\xFF") || line.find('\0')!=std::string::npos)) {
                    logger::error("SkyrimPrecombines.ini is saved as UTF-16 and cannot be read; save it as UTF-8 (Notepad: Save as, Encoding UTF-8). Default settings");
                    return json::object();
                }
                auto t=Trim(line);
                if(number==1 && t.starts_with("\xEF\xBB\xBF")) t=Trim(t.substr(3));  // UTF-8 byte order mark
                if(t.empty() || t.front()==';' || t.front()=='#' || t.front()=='[') continue;
                const auto equals=t.find('=');
                const auto name=equals==std::string_view::npos ? std::string() : Lower(Trim(t.substr(0,equals)));
                if(name.empty()) { logger::warn("SkyrimPrecombines.ini line {}: not a setting; ignored",number); continue; }
                const auto text=Trim(t.substr(equals+1));
                const auto shipped=std::find_if(kNames.begin(),kNames.end(),[&](const Name& n) { return n.ini==name; });
                if(shipped!=kNames.end()) {
                    auto value=Typed(name.front(),name.front()=='s' ? text : Plain(text));
                    if(!value) { logger::warn("SkyrimPrecombines.ini line {}: {} cannot be read; its default is used",number,Trim(t.substr(0,equals))); continue; }
                    config[std::string(shipped->key)]=std::move(*value);
                } else if(std::any_of(name.begin(),name.end(),[](char c) { return c=='_'; }) || name=="enabled" || name=="previs") {
                    // An internal name: its value as JSON, else as text.
                    auto value=json::parse(text,nullptr,false);
                    config[name]=value.is_discarded() ? json(std::string(text)) : std::move(value);
                } else { logger::warn("SkyrimPrecombines.ini line {}: unknown setting {}; ignored",number,Trim(t.substr(0,equals))); continue; }
                ++values;
            }
            logger::info("SkyrimPrecombines.ini: {} settings read",values);
            return config;
        }

        // A value as the log shows it; text that is not UTF-8 (a file saved in the ANSI code page) must not throw here.
        std::string Shown(const json& value) { return value.dump(-1,' ',false,json::error_handler_t::replace); }
        // Each setting that cannot be read is reported once, however often it is read.
        std::mutex reportMutex;
        std::set<std::string,std::less<>> reported;
        bool FirstReport(std::string_view key) {
            std::lock_guard lock(reportMutex);
            return reported.emplace(key).second;
        }
        template<class T> T Unreadable(std::string_view key,const json& value,std::string_view type,T fallback) {
            if(FirstReport(key)) logger::warn("SkyrimPrecombines.ini: {}={} is not {}; its default {} is used",key,Shown(value),type,fallback);
            return fallback;
        }
        // The readers, on any settings object (ConfigTest reads its cases with these).
        bool FlagIn(const json& config,std::string_view key,bool fallback) {
            const auto it=config.find(std::string(key));
            if(it==config.end()) return fallback;
            if(it->is_boolean()) return it->get<bool>();
            if(it->is_number_integer()) {
                const auto value=it->get<int64_t>();
                if(value==0 || value==1) return value==1;
            }
            return Unreadable(key,*it,"0, 1, true or false",fallback);
        }
        template<class T> T NumberIn(const json& config,std::string_view key,T fallback) {
            const auto it=config.find(std::string(key));
            if(it==config.end()) return fallback;
            if(it->is_number()) {
                const double value=it->get<double>();
                if(std::isfinite(value) && value>=static_cast<double>(std::numeric_limits<T>::lowest()) && value<=static_cast<double>(std::numeric_limits<T>::max()))
                    return it->get<T>();
            }
            return Unreadable(key,*it,std::is_floating_point_v<T> ? "a number" : std::is_signed_v<T> ? "a whole number" : "a whole number of 0 or more",fallback);
        }
        std::string TextIn(const json& config,std::string_view key,std::string fallback) {
            const auto it=config.find(std::string(key));
            if(it==config.end()) return fallback;
            if(it->is_string()) return it->get<std::string>();
            return Unreadable(key,*it,"text",std::move(fallback));
        }
        std::vector<uint32_t> FormIdsIn(const json& config,std::string_view key) {
            std::vector<uint32_t> ids;
            const auto it=config.find(std::string(key));
            if(it==config.end()) return ids;
            const auto hex=[&](std::string_view text) {
                text=Trim(text);
                if(text.starts_with("0x") || text.starts_with("0X")) text.remove_prefix(2);
                uint32_t id=0;
                const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),id,16);
                if(text.empty() || error!=std::errc() || end!=text.data()+text.size()) return false;
                ids.push_back(id);
                return true;
            };
            const auto add=[&](const json& item) {
                if(item.is_number()) {
                    const double value=item.get<double>();
                    if(value>=0.0 && value<=static_cast<double>(std::numeric_limits<uint32_t>::max())) {
                        // JSON reads 21000801 as a decimal number: a form ID copied from the console (8 hex digits) with no
                        // letter and no leading zero names another cell. Reported, once a key as every problem here;
                        // decimal lists ([86698]) stay as they are.
                        if(value>=1e7 && value<1e8 && FirstReport(key))
                            logger::warn("SkyrimPrecombines.ini: {}: {} is read as a decimal number (hex {:08X}); write a hex form ID as 0x{}",key,Shown(item),item.get<uint32_t>(),Shown(item));
                        ids.push_back(item.get<uint32_t>());
                        return;
                    }
                } else if(item.is_string() && hex(item.get_ref<const std::string&>())) return;
                if(FirstReport(key)) logger::warn("SkyrimPrecombines.ini: {}: {} is not a form ID (a number, or hex such as 000152AA); skipped",key,Shown(item));
            };
            if(it->is_array()) for(const auto& item:*it) add(item);
            else if(it->is_string() && it->get_ref<const std::string&>().find_first_of("[], \t")!=std::string::npos) {
                // Text that is not JSON (Read keeps it as written), such as [000152AA, 0001A26F]: several hex form IDs,
                // split at brackets, commas, spaces and quotes.
                const auto& text=it->get_ref<const std::string&>();
                for(size_t at=0;at<text.size();) {
                    const auto begin=text.find_first_not_of("[], \t\"",at);
                    if(begin==std::string::npos) break;
                    const auto end=std::min(text.find_first_of("[], \t\"",begin),text.size());
                    const auto token=std::string_view(text).substr(begin,end-begin);
                    if(!hex(token)) add(json(std::string(token)));
                    at=end;
                }
            } else add(*it);
            return ids;
        }

        // A path as UTF-8 for the log: path::string() gives the ANSI code page's bytes and throws for a character it lacks.
        std::string Utf8(const fs::path& path) { const auto text=path.u8string(); return std::string(text.begin(),text.end()); }

        // Writing changes back (the SKSE Menu Framework page), with the line rules of Read(), so what is written reads back
        // as set.
        constexpr std::string_view kBom = "\xEF\xBB\xBF";

        // A file Read() refuses as UTF-16 (its first line): never rewritten, which would mix two encodings in it.
        bool Utf16(std::string_view text) {
            const auto first=text.substr(0,text.find('\n'));
            return first.starts_with("\xFF\xFE") || first.starts_with("\xFE\xFF") || first.find('\0')!=std::string_view::npos;
        }
        // The shortest text that reads back as the same float, with a decimal point like the shipped file's (5.0, 0.01).
        std::optional<std::string> FloatText(double value) {
            if(!std::isfinite(value) || std::abs(value)>std::numeric_limits<float>::max()) return std::nullopt;
            char buffer[32];
            const auto [end,error]=std::to_chars(buffer,buffer+sizeof(buffer),static_cast<float>(value));
            if(error!=std::errc()) return std::nullopt;
            std::string text(buffer,end);
            if(text.find_first_of(".en")==std::string::npos) text+=".0";
            return text;
        }
        // A value the way an internal name's line holds it: JSON (true, 68, 7.5, "C:\\x").
        std::optional<std::string> JsonText(const json& value) {
            if(value.is_number_float()) return FloatText(value.get<double>());
            if(!value.is_boolean() && !value.is_number() && !value.is_string()) return std::nullopt;
            return Shown(value);
        }
        // A value the way Typed() reads a shipped name of this prefix: b 1/0, f and i a number, s the text itself, quoted
        // when Read() would otherwise drop its outer spaces or quotes. Nothing for a value of another type.
        std::optional<std::string> TypedText(char prefix,const json& value) {
            switch(prefix) {
            case 'b':
                if(value.is_boolean()) return value.get<bool>() ? "1" : "0";
                if(value.is_number()) return value.get<double>()!=0.0 ? "1" : "0";
                return std::nullopt;
            case 'f': case 'i':
                if(prefix=='i' && value.is_number_integer()) return value.dump();
                if(value.is_number()) return FloatText(value.get<double>());
                return std::nullopt;
            case 's': {
                if(!value.is_string()) return std::nullopt;
                const auto& text=value.get_ref<const std::string&>();
                if(text.find_first_of("\r\n")!=std::string::npos) return std::nullopt;
                const bool quote=!text.empty() && (Trim(text).size()!=text.size() || (text.front()=='"' && text.back()=='"'));
                return quote ? '"'+text+'"' : text;
            }
            }
            return std::nullopt;
        }

        struct Line { std::string text, end; };  // a line without its line ending, and that ending ("\r\n", "\n" or none)

        // A section header's name (lower case), or nothing for any other line.
        std::optional<std::string> Section(std::string_view line) {
            const auto t=Trim(line);
            if(!t.starts_with('[')) return std::nullopt;
            const auto inside=t.substr(1);
            return Lower(Trim(inside.substr(0,inside.find(']'))));
        }
        // The name a line sets (lower case, as Read() takes it), or empty: blank lines, comments and headers set none.
        std::string Setting(std::string_view line) {
            const auto t=Trim(line);
            if(t.empty() || t.front()==';' || t.front()=='#' || t.front()=='[') return {};
            const auto equals=t.find('=');
            return equals==std::string_view::npos ? std::string() : Lower(Trim(t.substr(0,equals)));
        }

        // Rewrite, also listing what it changed ("bHiZ=0") for the log.
        std::string Apply(std::string_view text,std::span<const Change> changes,std::vector<std::string>* changed) {
            const auto original=text;
            if(Utf16(text)) return std::string(text);
            const bool bom=text.starts_with(kBom);
            if(bom) text.remove_prefix(kBom.size());
            // Added lines end like the file's; a new file gets Notepad's (and the shipped file's) "\r\n".
            const std::string eol=text.empty() || text.find("\r\n")!=std::string_view::npos ? "\r\n" : "\n";
            std::vector<Line> lines;
            for(size_t start=0;start<text.size();) {
                const auto newline=text.find('\n',start);
                if(newline==std::string_view::npos) { lines.push_back({std::string(text.substr(start)),""}); break; }
                auto line=text.substr(start,newline-start);
                const bool crlf=line.ends_with('\r');
                if(crlf) line.remove_suffix(1);
                lines.push_back({std::string(line),crlf ? "\r\n" : "\n"});
                start=newline+1;
            }
            bool edited=false;
            for(const auto& change:changes) {
                auto key=Lower(Trim(change.key));
                if(key.empty() || key.find_first_of("=\r\n")!=std::string::npos || key.front()==';' || key.front()=='#' || key.front()=='[') continue;
                const auto shipped=std::find_if(kNames.begin(),kNames.end(),[&](const Name& n) { return n.key==key || n.ini==key; });
                const Name* name=shipped!=kNames.end() ? &*shipped : nullptr;
                if(name) key=name->key;
                // The last line that sets it is the one Read() keeps; it is changed in the spelling it uses.
                std::optional<size_t> last;
                bool lastShipped=false;
                for(size_t i=0;i<lines.size();++i) {
                    const auto setting=Setting(lines[i].text);
                    if(setting.empty()) continue;
                    if(setting==key) { last=i; lastShipped=false; }
                    else if(name && setting==name->ini) { last=i; lastShipped=true; }
                }
                const auto label=std::string(name ? name->display : std::string_view(key));
                if(last) {
                    auto& line=lines[*last];
                    const auto value=lastShipped ? TypedText(name->ini.front(),change.value) : JsonText(change.value);
                    if(!value) continue;
                    // The name, the '=' and the spaces after it stay, and on a b, f or i line also the quotes and the comment
                    // after the value, which Plain() reads past; the old value goes, and on other lines all that follows it.
                    auto at=line.text.find('=')+1;
                    while(at<line.text.size() && (line.text[at]==' ' || line.text[at]=='\t')) ++at;
                    const auto rest=std::string_view(line.text).substr(at);
                    auto updated=line.text.substr(0,at);
                    if(lastShipped && name->ini.front()!='s') {
                        const auto comment=std::min(rest.find_first_of(";#"),rest.size());
                        const auto old=rest.substr(0,comment),plain=Trim(old);
                        const bool quoted=plain.size()>=2 && plain.front()=='"' && plain.back()=='"';
                        updated+=quoted ? '"'+*value+'"' : *value;
                        if(comment<rest.size()) updated.append(old.substr(std::min(old.find_last_not_of(" \t")+1,old.size()))).append(rest.substr(comment));
                    } else updated+=*value;
                    if(updated==line.text) continue;
                    line.text=std::move(updated);
                    if(changed) changed->push_back(label+"="+*value);
                } else {
                    const auto value=name ? TypedText(name->ini.front(),change.value) : JsonText(change.value);
                    if(!value) continue;
                    const std::string section(name ? name->section : "Developer");
                    const auto lowerSection=Lower(section);
                    Line added{label+"="+*value,eol};
                    // Under the section's (last) header: after its last setting and the lines right below that one (its
                    // comments), or in a section without settings after the header and the lines right below it. A blank
                    // line ends those: comments after one belong to what follows, such as the next section's header.
                    std::optional<size_t> header;
                    for(size_t i=0;i<lines.size();++i) if(Section(lines[i].text)==lowerSection) header=i;
                    if(header) {
                        auto at=*header;
                        for(auto i=at+1;i<lines.size() && !Section(lines[i].text);++i) if(!Setting(lines[i].text).empty()) at=i;
                        while(at+1<lines.size() && !Section(lines[at+1].text) && !Trim(lines[at+1].text).empty()) ++at;
                        if(lines[at].end.empty()) lines[at].end=eol;
                        lines.insert(lines.begin()+static_cast<std::ptrdiff_t>(at+1),std::move(added));
                    } else {
                        // A new section at the end, after one empty line.
                        if(!lines.empty()) {
                            if(lines.back().end.empty()) lines.back().end=eol;
                            if(!Trim(lines.back().text).empty()) lines.push_back({"",eol});
                        }
                        lines.push_back({"["+section+"]",eol});
                        lines.push_back(std::move(added));
                    }
                    if(changed) changed->push_back(label+"="+*value);
                }
                edited=true;
            }
            if(!edited) return std::string(original);
            std::string out(bom ? kBom : std::string_view());
            for(const auto& line:lines) out.append(line.text).append(line.end);
            if(!lines.empty() && lines.back().end.empty()) out+=eol;
            return out;
        }

        // How writing a file ended: in full, not at all (it could not be opened, so it is as it was), or cut short.
        enum class Put { kDone, kNotOpened, kPartial };
        Put PutFile(const fs::path& target,std::string_view bytes) {
            std::ofstream out(target,std::ios::binary|std::ios::trunc);
            if(!out.is_open()) return Put::kNotOpened;
            out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
            out.close();
            return out.fail() ? Put::kPartial : Put::kDone;
        }
        using Writer=Put(*)(const fs::path&,std::string_view);

        // Replaces the file through <path>.tmp and a rename, so a write that fails halfway leaves the old file whole. If
        // only the rename fails (the file held open by an editor, some virtual file systems), the file is written in place.
        // `cut`: the file is cut short, by an earlier write in place that stopped partway or by this one; the .tmp, which
        // then holds the whole new text, is kept and named in the reason instead of removed.
        bool Replace(const fs::path& path,std::string_view bytes,bool& cut,std::string& reason,Writer put) {
            std::error_code ec;
            if(const auto folder=path.parent_path(); !folder.empty() && !fs::exists(folder,ec)) fs::create_directories(folder,ec);
            auto tmp=path;
            tmp+=".tmp";
            fs::remove(tmp,ec);
            if(put(tmp,bytes)!=Put::kDone) {
                fs::remove(tmp,ec);
                reason="a temporary file cannot be written next to it";
                return false;
            }
            fs::rename(tmp,path,ec);  // replaces the old file (MoveFileEx with MOVEFILE_REPLACE_EXISTING)
            if(!ec) { cut=false; return true; }
            const auto direct=put(path,bytes);
            if(direct==Put::kDone) {
                cut=false;
                fs::remove(tmp,ec);
                return true;
            }
            cut=cut || direct==Put::kPartial;
            if(cut) reason="it was cut short while being written; the whole new text is in "+tmp.filename().string();
            else {
                fs::remove(tmp,ec);  // not opened: the file is as it was
                reason="it cannot be replaced or written";
            }
            return false;
        }

        // While the file is cut short (Replace), the whole text it should hold: the next write starts from it rather than
        // from the cut file, so a retry loses no line. One write at a time.
        std::mutex writeMutex;
        std::optional<std::string> unsaved;

        // Write, with the file writer as a parameter (ConfigTest has a write in place stop partway).
        bool WriteWith(std::span<const Change> changes,Writer put) {
            std::lock_guard lock(writeMutex);
            std::string reason;
            try {
                const fs::path path(kPath);
                std::string text;
                std::error_code ec;
                if(unsaved) text=*unsaved;
                // A file that is there but cannot be read is left alone: never replaced by the changes alone.
                else if(fs::exists(path,ec)) {
                    std::ifstream in(path,std::ios::binary);
                    if(in) text.assign(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
                    if(!in.is_open() || in.bad()) reason="it cannot be read";
                    else if(Utf16(text)) reason="it is saved as UTF-16; save it as UTF-8";
                } else if(ec) reason=ec.message();
                if(reason.empty()) {
                    std::vector<std::string> changed;
                    const auto updated=Apply(text,changes,&changed);
                    if(updated==text && !unsaved) return true;
                    bool cut=unsaved.has_value();
                    const bool saved=Replace(path,updated,cut,reason,put);
                    if(cut) unsaved=updated;
                    else unsaved.reset();
                    if(saved) {
                        std::string list;
                        for(const auto& item:changed) list+=(list.empty() ? "" : ", ")+item;
                        if(list.empty()) logger::info("SkyrimPrecombines.ini saved in full again");
                        else logger::info("SkyrimPrecombines.ini saved: {}",list);
                        return true;
                    }
                }
            } catch(const std::exception& e) {
                reason=e.what();
            } catch(...) {
                reason="unknown error";
            }
            logger::warn("SkyrimPrecombines.ini could not be saved ({}); the change lasts until the game is closed",reason);
            return false;
        }
    }

    const json& Get() {
        static const json config=Read();
        return config;
    }

    bool Flag(std::string_view key,bool fallback) { return FlagIn(Get(),key,fallback); }
    template<class T> T Number(std::string_view key,T fallback) { return NumberIn(Get(),key,fallback); }
    template uint32_t Number<uint32_t>(std::string_view,uint32_t);
    template int Number<int>(std::string_view,int);
    template float Number<float>(std::string_view,float);
    template double Number<double>(std::string_view,double);
    std::string Text(std::string_view key,std::string fallback) { return TextIn(Get(),key,std::move(fallback)); }
    std::vector<uint32_t> FormIds(std::string_view key) { return FormIdsIn(Get(),key); }

    std::string Rewrite(std::string_view text,std::span<const Change> changes) { return Apply(text,changes,nullptr); }

    bool Write(std::span<const Change> changes) { return WriteWith(changes,PutFile); }

    const fs::path& CacheFolder() {
        static const fs::path folder=[] {
            const auto it=Get().find("cache_folder");
            if(it==Get().end() || !it->is_string() || it->get<std::string>().empty()) return fs::path();
            const auto& text=it->get_ref<const std::string&>();
            try {
                // The file is UTF-8 (Notepad's default), and so is every path the bake writes into JSON (ANSI text
                // failed every bake for a folder outside ASCII); its readers open files by wide path.
                auto path=fs::absolute(fs::path(std::u8string(text.begin(),text.end())));
                fs::create_directories(path/"RuntimeBake");
                logger::info("Runtime-baked cells are kept in {}",Utf8(path));
                return path;
            } catch(const std::exception& e) {
                logger::error("sCacheFolder {} is not usable ({}); runtime-baked cells are kept under Data",text,e.what());
                return fs::path();
            }
        }();
        return folder;
    }

    fs::path BakeDir() {
        const auto& folder=CacheFolder();
        return folder.empty() ? fs::path("Data/SKSE/Plugins/Precombines/RuntimeBake") : folder/"RuntimeBake";
    }

    fs::path MeshRoot() {
        const auto& folder=CacheFolder();
        return folder.empty() ? fs::path("Data/meshes") : folder/"meshes";
    }
}
