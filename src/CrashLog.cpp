#include "CrashLog.h"

#include <atomic>
#include <cinttypes>
#include <string>

// Windows Error Reporting can record nothing for a crash (a game on a private desktop, for example), so
// fatal-looking exceptions are logged here first: code, faulting instruction as module+offset, the accessed address,
// registers and return-address candidates from the stack. The handler only observes; it always continues the search.
namespace CrashLog {
    namespace {
        std::atomic<int> logged{0};

        std::string Describe(std::uintptr_t address) {
            HMODULE module{};
            if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCWSTR>(address),&module) || !module)
                return std::format("{:#x}",address);
            wchar_t path[MAX_PATH]{};
            GetModuleFileNameW(module,path,MAX_PATH);
            std::wstring wide(path);
            const auto slash=wide.find_last_of(L"\\/");
            std::string name;
            for(wchar_t c:wide.substr(slash==std::wstring::npos ? 0:slash+1)) name.push_back(c<128 ? static_cast<char>(c):'?');
            return std::format("{}+{:#x}",name,address-reinterpret_cast<std::uintptr_t>(module));
        }

        bool Executable(std::uintptr_t address) {
            MEMORY_BASIC_INFORMATION info{};
            if(!VirtualQuery(reinterpret_cast<LPCVOID>(address),&info,sizeof(info)) || info.State!=MEM_COMMIT) return false;
            return (info.Protect&(PAGE_EXECUTE|PAGE_EXECUTE_READ|PAGE_EXECUTE_READWRITE|PAGE_EXECUTE_WRITECOPY))!=0 && info.Type==MEM_IMAGE;
        }

        LONG CALLBACK Handler(EXCEPTION_POINTERS* info) {
            const auto* record=info->ExceptionRecord;
            switch(record->ExceptionCode) {
            case EXCEPTION_ACCESS_VIOLATION: case EXCEPTION_ILLEGAL_INSTRUCTION: case EXCEPTION_PRIV_INSTRUCTION:
            case EXCEPTION_INT_DIVIDE_BY_ZERO: case EXCEPTION_STACK_OVERFLOW: case EXCEPTION_IN_PAGE_ERROR:
            case 0xC0000409:  // fail-fast / stack buffer overrun
                break;
            default:
                return EXCEPTION_CONTINUE_SEARCH;
            }
            if(logged.fetch_add(1)>=8) return EXCEPTION_CONTINUE_SEARCH;
            const auto* context=info->ContextRecord;
            const auto rip=static_cast<std::uintptr_t>(context->Rip);
            std::string access;
            if(record->ExceptionCode==EXCEPTION_ACCESS_VIOLATION && record->NumberParameters>=2)
                access=std::format(" ({} {:#x})",record->ExceptionInformation[0]==0 ? "read":record->ExceptionInformation[0]==1 ? "write":"execute",
                                   record->ExceptionInformation[1]);
            logger::critical("Exception {:#010x} on thread {} at {}{}",static_cast<std::uint32_t>(record->ExceptionCode),GetCurrentThreadId(),Describe(rip),access);
            logger::critical("  rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} rbp={:#x} rsp={:#x}",
                context->Rax,context->Rbx,context->Rcx,context->Rdx,context->Rsi,context->Rdi,context->Rbp,context->Rsp);
            logger::critical("  r8={:#x} r9={:#x} r10={:#x} r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x}",
                context->R8,context->R9,context->R10,context->R11,context->R12,context->R13,context->R14,context->R15);
            if(record->ExceptionCode!=EXCEPTION_STACK_OVERFLOW) {
                const auto* tib=reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
                const auto low=reinterpret_cast<std::uintptr_t>(tib->StackLimit),high=reinterpret_cast<std::uintptr_t>(tib->StackBase);
                const auto start=static_cast<std::uintptr_t>(context->Rsp);
                const auto end=std::min<std::uintptr_t>(high,start+0x3000);
                int found=0;
                for(auto slot=start;slot+8<=end && found<24;slot+=8) {
                    if(slot<low) continue;
                    const auto value=*reinterpret_cast<const std::uintptr_t*>(slot);
                    if(Executable(value)) logger::critical("  stack[rsp+{:#x}] {}",slot-start,Describe(value)),++found;
                }
            }
            if(auto log=spdlog::default_logger()) log->flush();
            return EXCEPTION_CONTINUE_SEARCH;
        }
    }

    void Install() {
        if(AddVectoredExceptionHandler(1,Handler)) logger::info("Crash logging installed");
    }
}
