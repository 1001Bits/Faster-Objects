// Faster Objects (SkyrimPrecombines): precombined meshes and Hi-Z occlusion culling for Skyrim.
// Copyright (C) 2026 Asciim0v
//
// This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with this program. If not, see
// <https://www.gnu.org/licenses/>.

#pragma once

#include <d3d11.h>

#include <cstdint>

// Diagnostics: the GPU time of one span of commands per frame, from timestamp queries read back several frames later
// without waiting (D3D11_ASYNC_GETDATA_DONOTFLUSH). A frame whose slot has not come back yet is not measured. One
// thread only (the render thread).
class GpuTimer {
public:
    void Begin(ID3D11Device* device,ID3D11DeviceContext* context) {
        Collect(context);
        open_=false;
        auto& f=frames_[next_];
        if(f.pending) return;
        if(!f.disjoint) {
            D3D11_QUERY_DESC desc{D3D11_QUERY_TIMESTAMP_DISJOINT,0};
            if(FAILED(device->CreateQuery(&desc,&f.disjoint))) { f.disjoint=nullptr; return; }
            desc.Query=D3D11_QUERY_TIMESTAMP;
            if(FAILED(device->CreateQuery(&desc,&f.begin)) || FAILED(device->CreateQuery(&desc,&f.end))) return;
        }
        if(!f.begin || !f.end) return;
        context->Begin(f.disjoint);
        context->End(f.begin);
        open_=true;
    }
    void End(ID3D11DeviceContext* context) {
        if(!open_) return;
        auto& f=frames_[next_];
        context->End(f.end);
        context->End(f.disjoint);
        f.pending=true;
        next_=(next_+1)%kFrames;
        open_=false;
    }
    // Milliseconds summed over the spans measured since the previous call, and their number.
    void Take(double& ms,uint32_t& spans) { ms=sum_; spans=count_; sum_=0.0; count_=0; }

private:
    static constexpr int kFrames=8;
    struct Frame { ID3D11Query* disjoint{}; ID3D11Query* begin{}; ID3D11Query* end{}; bool pending=false; };
    void Collect(ID3D11DeviceContext* context) {
        for(auto& f:frames_) {
            if(!f.pending) continue;
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint{};
            UINT64 begin=0,end=0;
            if(context->GetData(f.disjoint,&disjoint,sizeof(disjoint),D3D11_ASYNC_GETDATA_DONOTFLUSH)!=S_OK ||
               context->GetData(f.begin,&begin,sizeof(begin),D3D11_ASYNC_GETDATA_DONOTFLUSH)!=S_OK ||
               context->GetData(f.end,&end,sizeof(end),D3D11_ASYNC_GETDATA_DONOTFLUSH)!=S_OK) continue;
            f.pending=false;
            if(disjoint.Disjoint || !disjoint.Frequency || end<begin) continue;
            sum_+=double(end-begin)*1000.0/double(disjoint.Frequency);
            ++count_;
        }
    }
    Frame frames_[kFrames];
    int next_=0;
    bool open_=false;
    double sum_=0.0;
    uint32_t count_=0;
};
