#include "HiZ.h"

#include <algorithm>
#include <xmmintrin.h>
#include <cmath>
#include <cstring>
#include <utility>

namespace HiZ {
    Mat4 Mat4::Identity() {
        Mat4 out;
        for(int i=0;i<4;++i) out.m[i][i]=1.f;
        return out;
    }

    Mat4 Multiply(const Mat4& a,const Mat4& b) {
        Mat4 out;
        for(int r=0;r<4;++r) for(int c=0;c<4;++c) {
            float sum=0.f;
            for(int k=0;k<4;++k) sum+=a.m[r][k]*b.m[k][c];
            out.m[r][c]=sum;
        }
        return out;
    }

    Mat4 Transpose(const Mat4& a) {
        Mat4 out;
        for(int r=0;r<4;++r) for(int c=0;c<4;++c) out.m[r][c]=a.m[c][r];
        return out;
    }

    Mat4d ToDouble(const Mat4& a) {
        Mat4d out;
        for(int r=0;r<4;++r) for(int c=0;c<4;++c) out.m[r][c]=a.m[r][c];
        return out;
    }

    Mat4 ToFloat(const Mat4d& a) {
        Mat4 out;
        for(int r=0;r<4;++r) for(int c=0;c<4;++c) out.m[r][c]=static_cast<float>(a.m[r][c]);
        return out;
    }

    Mat4d Multiply(const Mat4d& a,const Mat4d& b) {
        Mat4d out;
        for(int r=0;r<4;++r) for(int c=0;c<4;++c) {
            double sum=0.0;
            for(int k=0;k<4;++k) sum+=a.m[r][k]*b.m[k][c];
            out.m[r][c]=sum;
        }
        return out;
    }

    // Gauss-Jordan elimination with partial pivoting, in double precision: world coordinates reach
    // hundreds of thousands of units, and the inverse feeds every reprojected texel.
    bool Invert(const Mat4d& in,Mat4d& out) {
        double a[4][8];
        for(int r=0;r<4;++r) for(int c=0;c<4;++c) { a[r][c]=in.m[r][c]; a[r][c+4]=r==c ? 1.0:0.0; }
        for(int col=0;col<4;++col) {
            int pivot=col;
            for(int r=col+1;r<4;++r) if(std::abs(a[r][col])>std::abs(a[pivot][col])) pivot=r;
            if(std::abs(a[pivot][col])<1e-12) return false;
            if(pivot!=col) for(int c=0;c<8;++c) std::swap(a[pivot][c],a[col][c]);
            const double scale=1.0/a[col][col];
            for(int c=0;c<8;++c) a[col][c]*=scale;
            for(int r=0;r<4;++r) {
                if(r==col) continue;
                const double factor=a[r][col];
                if(factor!=0.0) for(int c=0;c<8;++c) a[r][c]-=factor*a[col][c];
            }
        }
        for(int r=0;r<4;++r) for(int c=0;c<4;++c) out.m[r][c]=a[r][c+4];
        return true;
    }

    bool Invert(const Mat4& in,Mat4& out) {
        Mat4d inverse;
        if(!Invert(ToDouble(in),inverse)) return false;
        out=ToFloat(inverse);
        return true;
    }

    bool Reprojection(const Mat4d& current,const Mat4d& captured,Mat4& out) {
        Mat4d inverse;
        if(!Invert(captured,inverse)) return false;
        const Mat4d r=Multiply(current,inverse);
        const bool same=std::memcmp(&current,&captured,sizeof(Mat4d))==0;
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) {
            if(!std::isfinite(r.m[i][j])) return false;
            if(same && std::abs(r.m[i][j]-(i==j ? 1.0:0.0))>1e-6) return false;
        }
        out=ToFloat(r);
        return true;
    }

    Mat4 Relative(const Mat4d& worldToClip,const std::array<float,3>& origin) {
        Mat4d out=worldToClip;
        const auto& m=worldToClip.m;
        for(int r=0;r<4;++r) out.m[r][3]=m[r][3]+m[r][0]*double(origin[0])+m[r][1]*double(origin[1])+m[r][2]*double(origin[2]);
        return ToFloat(out);
    }

    bool ScreenRect(const Mat4& worldToClip,const float c[3],float radius,float rect[4]) {
        if(!(radius>=0.f)) return false;
        const auto& m=worldToClip.m;
        float norm[4];
        for(int r=0;r<4;++r) norm[r]=std::sqrt(m[r][0]*m[r][0]+m[r][1]*m[r][1]+m[r][2]*m[r][2]);
        const float cx=m[0][0]*c[0]+m[0][1]*c[1]+m[0][2]*c[2]+m[0][3];
        const float cy=m[1][0]*c[0]+m[1][1]*c[1]+m[1][2]*c[2]+m[1][3];
        const float cw=m[3][0]*c[0]+m[3][1]*c[1]+m[3][2]*c[2]+m[3][3];
        // Over the sphere each clip component varies by at most radius times its row length (as in Pyramid::Occluded).
        const float wmin=cw-radius*norm[3],wmax=cw+radius*norm[3];
        if(!(wmin>1.f)) return false;
        const auto range=[&](float v,float r,float& lo,float& hi) {
            const float a=(v-r)/wmin,b=(v-r)/wmax,d=(v+r)/wmin,e=(v+r)/wmax;
            lo=std::min({a,b,d,e}); hi=std::max({a,b,d,e});
        };
        range(cx,radius*norm[0],rect[0],rect[2]);
        range(cy,radius*norm[1],rect[1],rect[3]);
        return true;
    }

    float FarthestW(const Mat4& worldToClip,const float c[3],float radius) {
        const auto& m=worldToClip.m;
        const float norm=std::sqrt(m[3][0]*m[3][0]+m[3][1]*m[3][1]+m[3][2]*m[3][2]);
        return m[3][0]*c[0]+m[3][1]*c[1]+m[3][2]*c[2]+m[3][3]+radius*norm;
    }

    Mat4 FromTransform(const float r[3][3],float scale,const float t[3]) {
        Mat4 out;
        for(int i=0;i<3;++i) {
            for(int j=0;j<3;++j) out.m[i][j]=r[i][j]*scale;
            out.m[i][3]=t[i];
        }
        out.m[3][3]=1.f;
        return out;
    }

    uint32_t Pyramid::Build(const Grid& grid,const Mat4& worldToClip,const DepthModel& depth,const std::array<float,3>& eye,float nearExempt,
                            const Mat4d* exact) {
        return Build(grid,worldToClip,depth,eye,nearExempt,[](const std::function<void()>& a,const std::function<void()>& b) { a(); b(); },exact);
    }

    uint32_t Pyramid::Build(const Grid& grid,const Mat4& worldToClip,const DepthModel& depth,const std::array<float,3>& eye,float nearExempt,
                            const Both& both,const Mat4d* exact) {
        clip_=worldToClip; depth_=depth; eye_=eye; nearExempt_=nearExempt;
        for(int r=0;r<4;++r) rowNorm_[r]=std::sqrt(clip_.m[r][0]*clip_.m[r][0]+clip_.m[r][1]*clip_.m[r][1]+clip_.m[r][2]*clip_.m[r][2]);
        ready_=false;
        const uint32_t w=grid.width,h=grid.height;
        // One matrix takes a captured texel (NDC x, y and depth) straight to the current clip space, composed in double
        // (see Mat4d).
        Mat4 r;
        const Mat4d current=exact ? *exact : ToDouble(worldToClip),captured=grid.hasExact ? grid.exact : ToDouble(grid.worldToClip);
        if(!w || !h || grid.depth.size()!=size_t(w)*h || !Reprojection(current,captured,r)) return 0;
        // A texel lands on the one texel its centre reaches, but it stands for about a texel's area there: after the camera
        // moved, a texel can receive only a near surface's sample while part of it shows a far surface whose sample landed
        // next door, and it would read nearer than all it shows (a small object beside an outline hidden for a frame).
        // A moved view therefore takes the farthest of
        // each texel's 3x3 neighbourhood, landing into a ring one texel outside the frame for the edge texels. That holds
        // while a captured texel covers under two texels here; one spread wider (a surface walked up to) leaves texels
        // between its samples that nothing reaches, which read as far and spread too. A still view lands every texel on
        // itself and keeps its exact grid.
        const bool moved=std::memcmp(&current,&captured,sizeof(Mat4d))!=0;
        const uint32_t ring=moved ? 1u : 0u,pw=w+2*ring,ph=h+2*ring;
        const float fring=static_cast<float>(ring);
        if(widths_.empty() || widths_[0]!=w || heights_[0]!=h) {
            levels_.clear(); widths_.clear(); heights_.clear();
            uint32_t lw=w,lh=h;
            while(true) {
                levels_.emplace_back(size_t(lw)*lh); widths_.push_back(lw); heights_.push_back(lh);
                if(lw==1 && lh==1) break;
                lw=std::max(1u,(lw+1)/2); lh=std::max(1u,(lh+1)/2);
            }
        }
        // Within a row, NDC y is constant, so each clip component is a*nx + b*d + c with per-row c; four texels per step.
        auto& landing=moved ? padded_ : levels_[0];
        landing.assign(size_t(pw)*ph,-1.f);
        scratch_.assign(size_t(pw)*ph,-1.f);
        const float fw=static_cast<float>(w),fh=static_cast<float>(h),step=2.f/fw;
        const auto rows=[&](uint32_t yBegin,uint32_t yEnd,std::vector<float>& level,uint32_t& landed) {
        const __m128 one=_mm_set1_ps(1.f),zero=_mm_setzero_ps(),half=_mm_set1_ps(.5f),wMin=_mm_set1_ps(1e-4f);
        const __m128 vfw=_mm_set1_ps(fw),vfh=_mm_set1_ps(fh),lanes=_mm_setr_ps(0.f,1.f,2.f,3.f),vstep=_mm_set1_ps(step);
        const __m128 edge=_mm_set1_ps(-fring),right=_mm_set1_ps(fw+fring),bottom=_mm_set1_ps(fh+fring);
        __m128 a[4],b[4];
        for(int k=0;k<4;++k) { a[k]=_mm_set1_ps(r.m[k][0]); b[k]=_mm_set1_ps(r.m[k][2]); }
        alignas(16) float zs[4],xs[4],ys[4];
        const auto scatter=[&](float z,float px,float py) {
            // Several texels can land on one: keep the farthest, which never hides more than before.
            float& target=level[size_t(py+fring)*pw+size_t(px+fring)];
            target=std::max(target,z);
            ++landed;
        };
        for(uint32_t y=yBegin;y<yEnd;++y) {
            const float ny=1.f-(y+.5f)*2.f/fh;
            const float* row=grid.depth.data()+size_t(y)*w;
            float rc[4];
            for(int k=0;k<4;++k) rc[k]=r.m[k][1]*ny+r.m[k][3];
            const __m128 c0=_mm_set1_ps(rc[0]),c1=_mm_set1_ps(rc[1]),c2=_mm_set1_ps(rc[2]),c3=_mm_set1_ps(rc[3]);
            uint32_t x=0;
            for(;x+4<=w;x+=4) {
                const __m128 d=_mm_loadu_ps(row+x);
                // Far plane, sky or no data (d >= 1, d < 0, NaN) occludes nothing.
                __m128 valid=_mm_and_ps(_mm_cmplt_ps(d,one),_mm_cmpge_ps(d,zero));
                if(!_mm_movemask_ps(valid)) continue;
                const __m128 nx=_mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_set1_ps(static_cast<float>(x)+.5f),lanes),vstep),one);
                const __m128 cw=_mm_add_ps(_mm_add_ps(_mm_mul_ps(a[3],nx),_mm_mul_ps(b[3],d)),c3);
                valid=_mm_and_ps(valid,_mm_cmpgt_ps(cw,wMin));
                const __m128 inv=_mm_div_ps(one,cw);
                const __m128 z=_mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(a[2],nx),_mm_mul_ps(b[2],d)),c2),inv);
                const __m128 px=_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(a[0],nx),_mm_mul_ps(b[0],d)),c0),inv),half),half),vfw);
                const __m128 py=_mm_mul_ps(_mm_sub_ps(half,_mm_mul_ps(_mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(a[1],nx),_mm_mul_ps(b[1],d)),c1),inv),half)),vfh);
                valid=_mm_and_ps(valid,_mm_and_ps(_mm_cmpge_ps(z,zero),_mm_cmplt_ps(z,one)));
                valid=_mm_and_ps(valid,_mm_and_ps(_mm_and_ps(_mm_cmpge_ps(px,edge),_mm_cmplt_ps(px,right)),
                                                  _mm_and_ps(_mm_cmpge_ps(py,edge),_mm_cmplt_ps(py,bottom))));
                const int mask=_mm_movemask_ps(valid);
                if(!mask) continue;
                _mm_store_ps(zs,z); _mm_store_ps(xs,px); _mm_store_ps(ys,py);
                for(int lane=0;lane<4;++lane) if(mask&(1<<lane)) scatter(zs[lane],xs[lane],ys[lane]);
            }
            for(;x<w;++x) {
                const float d=row[x];
                if(!(d<1.f) || d<0.f) continue;
                const float nx=(x+.5f)*step-1.f;
                const float cw=r.m[3][0]*nx+r.m[3][2]*d+rc[3];
                if(!(cw>1e-4f)) continue;
                const float inv=1.f/cw;
                const float z=(r.m[2][0]*nx+r.m[2][2]*d+rc[2])*inv;
                if(!(z>=0.f && z<1.f)) continue;
                const float px=((r.m[0][0]*nx+r.m[0][2]*d+rc[0])*inv*.5f+.5f)*fw;
                const float py=(.5f-(r.m[1][0]*nx+r.m[1][2]*d+rc[1])*inv*.5f)*fh;
                if(!(px>=-fring && py>=-fring && px<fw+fring && py<fh+fring)) continue;
                scatter(z,px,py);
            }
        }
        };
        uint32_t landedFirst=0,landedSecond=0;
        const uint32_t middle=h/2;
        both([&] { rows(0,middle,landing,landedFirst); },[&] { rows(middle,h,scratch_,landedSecond); });
        for(size_t i=0;i<landing.size();++i) landing[i]=std::max(landing[i],scratch_[i]);
        const uint32_t landed=landedFirst+landedSecond;
        for(auto& value:landing) if(value<0.f) value=1.f;
        if(moved) {
            // The 3x3 farthest in two passes: across each padded row into scratch_ (now free), then down into level 0.
            for(uint32_t y=0;y<ph;++y) {
                const float* in=landing.data()+size_t(y)*pw;
                float* out=scratch_.data()+size_t(y)*w;
                for(uint32_t x=0;x<w;++x) out[x]=std::max(std::max(in[x],in[x+1]),in[x+2]);
            }
            auto& level=levels_[0];
            for(uint32_t y=0;y<h;++y) {
                const float* above=scratch_.data()+size_t(y)*w;
                for(uint32_t x=0;x<w;++x) level[size_t(y)*w+x]=std::max(std::max(above[x],above[x+w]),above[x+2*size_t(w)]);
            }
        }
        for(size_t i=1;i<levels_.size();++i) {
            const uint32_t pw=widths_[i-1],ph=heights_[i-1],nw=widths_[i],nh=heights_[i];
            const auto& prev=levels_[i-1];
            auto& next=levels_[i];
            for(uint32_t y=0;y<nh;++y) {
                const uint32_t y0=std::min(2*y,ph-1),y1=std::min(2*y+1,ph-1);
                for(uint32_t x=0;x<nw;++x) {
                    const uint32_t x0=std::min(2*x,pw-1),x1=std::min(2*x+1,pw-1);
                    next[size_t(y)*nw+x]=std::max(std::max(prev[size_t(y0)*pw+x0],prev[size_t(y0)*pw+x1]),
                                                  std::max(prev[size_t(y1)*pw+x0],prev[size_t(y1)*pw+x1]));
                }
            }
        }
        ready_=true;
        return landed;
    }

    float Pyramid::MaxOver(size_t level,uint32_t x0,uint32_t y0,uint32_t x1,uint32_t y1) const {
        const auto& data=levels_[level];
        const uint32_t w=widths_[level];
        float out=0.f;
        for(uint32_t y=y0;y<=y1;++y) for(uint32_t x=x0;x<=x1;++x) out=std::max(out,data[size_t(y)*w+x]);
        return out;
    }

    bool Pyramid::Occluded(const float c[3],float radius) const {
        if(!ready_ || !(radius>=0.f)) return false;
        const float ex=c[0]-eye_[0],ey=c[1]-eye_[1],ez=c[2]-eye_[2];
        if(std::sqrt(ex*ex+ey*ey+ez*ez)-radius<nearExempt_) return false;
        const auto& m=clip_.m;
        const float cx=m[0][0]*c[0]+m[0][1]*c[1]+m[0][2]*c[2]+m[0][3];
        const float cy=m[1][0]*c[0]+m[1][1]*c[1]+m[1][2]*c[2]+m[1][3];
        const float cw=m[3][0]*c[0]+m[3][1]*c[1]+m[3][2]*c[2]+m[3][3];
        // Over the sphere each clip component varies by at most radius times its row length.
        const float xr=radius*rowNorm_[0],yr=radius*rowNorm_[1],wr=radius*rowNorm_[3];
        const float wmin=cw-wr,wmax=cw+wr;
        if(!(wmin>1.f)) return false;  // reaches the near plane or lies behind the eye
        // The extremes of (v -/+ r) / w over w in [wmin, wmax] (wmin > 1, r >= 0): of the four quotients the smallest
        // has the lower numerator and, when that is negative, the smaller divisor (division is monotonic, so these are
        // the same floats OccludedReference picks with min and max).
        const auto range=[&](float v,float r,float& lo,float& hi) {
            const float a=v-r,b=v+r;
            lo=a>=0.f ? a/wmax : a/wmin;
            hi=b>=0.f ? b/wmin : b/wmax;
        };
        float x0,x1,y0,y1;
        range(cx,xr,x0,x1); range(cy,yr,y0,y1);
        if(x1<-1.f || x0>1.f || y1<-1.f || y0>1.f) return false;  // off screen: frustum culling decides
        x0=std::max(x0,-1.f); x1=std::min(x1,1.f); y0=std::max(y0,-1.f); y1=std::min(y1,1.f);
        const uint32_t w=widths_[0],h=heights_[0];
        const auto texel=[](float v,uint32_t size) { return std::min(size-1,static_cast<uint32_t>(std::max(0.f,v*size))); };
        uint32_t tx0=texel(x0*.5f+.5f,w),tx1=texel(x1*.5f+.5f,w);
        uint32_t ty0=texel(.5f-y1*.5f,h),ty1=texel(.5f-y0*.5f,h);
        size_t level=0;
        while(level+1<levels_.size() && ((tx1>>level)-(tx0>>level)>=4 || (ty1>>level)-(ty0>>level)>=4)) ++level;
        const float occluder=MaxOver(level,tx0>>level,ty0>>level,tx1>>level,ty1>>level);
        if(!(occluder<depth_.a) || occluder>=1.f) return false;  // something in the rectangle is far or unknown
        // Compare in view depth: NDC depth is too compressed near the far plane for a fixed tolerance.
        return wmin>depth_.ViewDepth(occluder)*1.001f+1.f;
    }

    bool Pyramid::OccludedReference(const float c[3],float radius) const {
        if(!ready_ || !(radius>=0.f)) return false;
        const float ex=c[0]-eye_[0],ey=c[1]-eye_[1],ez=c[2]-eye_[2];
        if(std::sqrt(ex*ex+ey*ey+ez*ez)-radius<nearExempt_) return false;
        const auto& m=clip_.m;
        const float cx=m[0][0]*c[0]+m[0][1]*c[1]+m[0][2]*c[2]+m[0][3];
        const float cy=m[1][0]*c[0]+m[1][1]*c[1]+m[1][2]*c[2]+m[1][3];
        const float cw=m[3][0]*c[0]+m[3][1]*c[1]+m[3][2]*c[2]+m[3][3];
        // Over the sphere each clip component varies by at most radius times its row length.
        const float xr=radius*rowNorm_[0],yr=radius*rowNorm_[1],wr=radius*rowNorm_[3];
        const float wmin=cw-wr,wmax=cw+wr;
        if(!(wmin>1.f)) return false;  // reaches the near plane or lies behind the eye
        const auto range=[&](float v,float r,float& lo,float& hi) {
            const float a=(v-r)/wmin,b=(v-r)/wmax,d=(v+r)/wmin,e=(v+r)/wmax;
            lo=std::min({a,b,d,e}); hi=std::max({a,b,d,e});
        };
        float x0,x1,y0,y1;
        range(cx,xr,x0,x1); range(cy,yr,y0,y1);
        if(x1<-1.f || x0>1.f || y1<-1.f || y0>1.f) return false;  // off screen: frustum culling decides
        x0=std::max(x0,-1.f); x1=std::min(x1,1.f); y0=std::max(y0,-1.f); y1=std::min(y1,1.f);
        const uint32_t w=widths_[0],h=heights_[0];
        const auto texel=[](float v,uint32_t size) { return std::min(size-1,static_cast<uint32_t>(std::max(0.f,v*size))); };
        uint32_t tx0=texel(x0*.5f+.5f,w),tx1=texel(x1*.5f+.5f,w);
        uint32_t ty0=texel(.5f-y1*.5f,h),ty1=texel(.5f-y0*.5f,h);
        size_t level=0;
        while(level+1<levels_.size() && ((tx1>>level)-(tx0>>level)>=4 || (ty1>>level)-(ty0>>level)>=4)) ++level;
        const float occluder=MaxOver(level,tx0>>level,ty0>>level,tx1>>level,ty1>>level);
        if(!(occluder<depth_.a) || occluder>=1.f) return false;  // something in the rectangle is far or unknown
        // Compare in view depth: NDC depth is too compressed near the far plane for a fixed tolerance.
        return wmin>depth_.ViewDepth(occluder)*1.001f+1.f;
    }

    Pyramid::Explanation Pyramid::Explain(const float c[3],float radius) const {
        Explanation out;
        if(!ready_ || !(radius>=0.f)) { out.why="no pyramid"; return out; }
        const float ex=c[0]-eye_[0],ey=c[1]-eye_[1],ez=c[2]-eye_[2];
        out.distance=std::sqrt(ex*ex+ey*ey+ez*ez);
        if(out.distance-radius<nearExempt_) { out.why="near the eye"; return out; }
        const auto& m=clip_.m;
        const float cx=m[0][0]*c[0]+m[0][1]*c[1]+m[0][2]*c[2]+m[0][3];
        const float cy=m[1][0]*c[0]+m[1][1]*c[1]+m[1][2]*c[2]+m[1][3];
        const float cw=m[3][0]*c[0]+m[3][1]*c[1]+m[3][2]*c[2]+m[3][3];
        const float xr=radius*rowNorm_[0],yr=radius*rowNorm_[1],wr=radius*rowNorm_[3];
        const float wmin=cw-wr,wmax=cw+wr;
        out.wmin=wmin;
        if(!(wmin>1.f)) { out.why="reaches the near plane"; return out; }
        const auto range=[&](float v,float r,float& lo,float& hi) {
            const float a=v-r,b=v+r;
            lo=a>=0.f ? a/wmax : a/wmin;
            hi=b>=0.f ? b/wmin : b/wmax;
        };
        float x0,x1,y0,y1;
        range(cx,xr,x0,x1); range(cy,yr,y0,y1);
        if(x1<-1.f || x0>1.f || y1<-1.f || y0>1.f) { out.why="off screen"; return out; }
        x0=std::max(x0,-1.f); x1=std::min(x1,1.f); y0=std::max(y0,-1.f); y1=std::min(y1,1.f);
        const uint32_t w=widths_[0],h=heights_[0];
        const auto texel=[](float v,uint32_t size) { return std::min(size-1,static_cast<uint32_t>(std::max(0.f,v*size))); };
        out.tx0=texel(x0*.5f+.5f,w); out.tx1=texel(x1*.5f+.5f,w);
        out.ty0=texel(.5f-y1*.5f,h); out.ty1=texel(.5f-y0*.5f,h);
        size_t level=0;
        while(level+1<levels_.size() && ((out.tx1>>level)-(out.tx0>>level)>=4 || (out.ty1>>level)-(out.ty0>>level)>=4)) ++level;
        out.level=static_cast<uint32_t>(level);
        out.occluder=MaxOver(level,out.tx0>>level,out.ty0>>level,out.tx1>>level,out.ty1>>level);
        if(!(out.occluder<depth_.a) || out.occluder>=1.f) { out.why="far or unknown depth in the rectangle"; return out; }
        out.occluderView=depth_.ViewDepth(out.occluder);
        out.hidden=wmin>out.occluderView*1.001f+1.f;
        out.why=out.hidden ? "every texel nearer than the sphere" : "a texel behind the sphere's nearest point";
        return out;
    }

    namespace {
        uint64_t Mix(const void* object) {
            uint64_t h=reinterpret_cast<uintptr_t>(object);
            h^=h>>33; h*=0xff51afd7ed558ccdull; h^=h>>33; h*=0xc4ceb9fe1a85ec53ull; h^=h>>33;
            return h;
        }
    }
    uint64_t VisibleCache::Hash(const void* object) { return Mix(object); }
    VisibleCache::VisibleCache(uint32_t slotsLog2) : slots_(new std::atomic<uint64_t>[size_t(1)<<slotsLog2]()),shift_(64-slotsLog2) {}
    bool VisibleCache::Skip(const void* object,uint32_t frame) const {
        const uint64_t h=Hash(object);
        const uint64_t entry=slots_[h>>shift_].load(std::memory_order_relaxed);
        return static_cast<uint32_t>(entry>>32)==(static_cast<uint32_t>(h)|1u) && static_cast<int32_t>(static_cast<uint32_t>(entry)-frame)>0;
    }
    void VisibleCache::Visible(const void* object,uint32_t frame,uint32_t interval) {
        if(interval<2) return;
        const uint64_t h=Hash(object);
        const uint32_t until=frame+1u+static_cast<uint32_t>((h>>40)%interval);
        slots_[h>>shift_].store(uint64_t(static_cast<uint32_t>(h)|1u)<<32|until,std::memory_order_relaxed);
    }

    ShownFrames::ShownFrames(uint32_t slotsLog2) : slots_(new std::atomic<uint64_t>[size_t(1)<<slotsLog2]()),shift_(64-slotsLog2) {}
    void ShownFrames::Shown(const void* object,uint32_t frame) {
        const uint64_t h=Mix(object);
        slots_[h>>shift_].store(uint64_t(static_cast<uint32_t>(h)|1u)<<32|frame,std::memory_order_relaxed);
    }
    uint32_t ShownFrames::Last(const void* object) const {
        const uint64_t h=Mix(object);
        const uint64_t entry=slots_[h>>shift_].load(std::memory_order_relaxed);
        return static_cast<uint32_t>(entry>>32)==(static_cast<uint32_t>(h)|1u) ? static_cast<uint32_t>(entry) : 0u;
    }

    bool YieldGate::Step(uint32_t tested,uint32_t culled) {
        if(!(minYield>0.f) || interval<2 || burst<1) return true;
        if(tested>=minTests) {
            const bool poor=static_cast<float>(culled)<minYield*static_cast<float>(tested);
            if(!probing) {
                low=poor ? low+1 : 0;
                if(low>=lowFrames) { probing=true; frame=0; }
            } else if(!poor) {
                probing=false; low=0;
            }
        }
        if(!probing) return true;
        return (frame++%interval)<burst;
    }
}
