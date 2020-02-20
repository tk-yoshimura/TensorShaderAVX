#include "../TensorShaderAvxBackend.h"

using namespace System;

void linear_zoom_3d(unsigned int channels,
    unsigned int inwidth, unsigned int inheight, unsigned int indepth,
    unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
    unsigned int th,
    const float* __restrict inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * inheight * indepth * th;
    const unsigned int outmap_offset = channels * outwidth * outheight * outdepth * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    const __m256 cinv27 = _mm256_set1_ps(float(1.0 / 27.0));
    const __m256 c8 = _mm256_set1_ps(8);
    const __m256 c4 = _mm256_set1_ps(4);
    const __m256 c2 = _mm256_set1_ps(2);

    const unsigned int luf = 0, ruf = channels, ldf = channels * outwidth, rdf = channels * (1 + outwidth);
    const unsigned int lur = channels * outwidth * outheight, rur = ruf + lur, ldr = ldf + lur, rdr = rdf + lur;

    for (unsigned int izc = 0, oz = 0; izc < indepth; izc++, oz += 2) {
        unsigned int izf = ((izc > 0) ? izc - 1 : 0);
        unsigned int izr = ((izc < indepth - 1) ? izc + 1 : indepth - 1);

        for (unsigned int iyc = 0, oy = 0; iyc < inheight; iyc++, oy += 2) {
            unsigned int iyu = ((iyc > 0) ? iyc - 1 : 0);
            unsigned int iyd = ((iyc < inheight - 1) ? iyc + 1 : inheight - 1);

            for (unsigned int ixc = 0, ox = 0; ixc < inwidth; ixc++, ox += 2) {
                unsigned int ixl = ((ixc > 0) ? ixc - 1 : 0);
                unsigned int ixr = ((ixc < inwidth - 1) ? ixc + 1 : inwidth - 1);

                unsigned int inmap_c_idx    = channels * (ixc + inwidth * (iyc + inheight * izc));
                unsigned int inmap_xl_idx   = channels * (ixl + inwidth * (iyc + inheight * izc));
                unsigned int inmap_xr_idx   = channels * (ixr + inwidth * (iyc + inheight * izc));
                unsigned int inmap_yu_idx   = channels * (ixc + inwidth * (iyu + inheight * izc));
                unsigned int inmap_yd_idx   = channels * (ixc + inwidth * (iyd + inheight * izc));
                unsigned int inmap_zf_idx   = channels * (ixc + inwidth * (iyc + inheight * izf));
                unsigned int inmap_zr_idx   = channels * (ixc + inwidth * (iyc + inheight * izr));
                unsigned int inmap_xlyu_idx = channels * (ixl + inwidth * (iyu + inheight * izc));
                unsigned int inmap_xryu_idx = channels * (ixr + inwidth * (iyu + inheight * izc));
                unsigned int inmap_xlyd_idx = channels * (ixl + inwidth * (iyd + inheight * izc));
                unsigned int inmap_xryd_idx = channels * (ixr + inwidth * (iyd + inheight * izc));
                unsigned int inmap_xlzf_idx = channels * (ixl + inwidth * (iyc + inheight * izf));
                unsigned int inmap_xrzf_idx = channels * (ixr + inwidth * (iyc + inheight * izf));
                unsigned int inmap_xlzr_idx = channels * (ixl + inwidth * (iyc + inheight * izr));
                unsigned int inmap_xrzr_idx = channels * (ixr + inwidth * (iyc + inheight * izr));
                unsigned int inmap_yuzf_idx = channels * (ixc + inwidth * (iyu + inheight * izf));
                unsigned int inmap_ydzf_idx = channels * (ixc + inwidth * (iyd + inheight * izf));
                unsigned int inmap_yuzr_idx = channels * (ixc + inwidth * (iyu + inheight * izr));
                unsigned int inmap_ydzr_idx = channels * (ixc + inwidth * (iyd + inheight * izr));
                unsigned int inmap_xlyuzf_idx = channels * (ixl + inwidth * (iyu + inheight * izf));
                unsigned int inmap_xryuzf_idx = channels * (ixr + inwidth * (iyu + inheight * izf));
                unsigned int inmap_xlydzf_idx = channels * (ixl + inwidth * (iyd + inheight * izf));
                unsigned int inmap_xrydzf_idx = channels * (ixr + inwidth * (iyd + inheight * izf));
                unsigned int inmap_xlyuzr_idx = channels * (ixl + inwidth * (iyu + inheight * izr));
                unsigned int inmap_xryuzr_idx = channels * (ixr + inwidth * (iyu + inheight * izr));
                unsigned int inmap_xlydzr_idx = channels * (ixl + inwidth * (iyd + inheight * izr));
                unsigned int inmap_xrydzr_idx = channels * (ixr + inwidth * (iyd + inheight * izr));

                unsigned int outmap_idx = channels * (ox + outwidth * (oy + outheight * oz));

                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 xc  = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_c_idx + i), c8);
                    
                    __m256 xxl = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xl_idx + i), c4);
                    __m256 xxr = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xr_idx + i), c4);
                    __m256 xyu = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_yu_idx + i), c4);
                    __m256 xyd = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_yd_idx + i), c4);
                    __m256 xzf = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_zf_idx + i), c4);
                    __m256 xzr = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_zr_idx + i), c4);

                    __m256 xxlyu = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xlyu_idx + i), c2);
                    __m256 xxryu = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xryu_idx + i), c2);
                    __m256 xxlyd = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xlyd_idx + i), c2);
                    __m256 xxryd = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xryd_idx + i), c2);
                    __m256 xxlzf = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xlzf_idx + i), c2);
                    __m256 xxrzf = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xrzf_idx + i), c2);
                    __m256 xxlzr = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xlzr_idx + i), c2);
                    __m256 xxrzr = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_xrzr_idx + i), c2);
                    __m256 xyuzf = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_yuzf_idx + i), c2);
                    __m256 xydzf = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_ydzf_idx + i), c2);
                    __m256 xyuzr = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_yuzr_idx + i), c2);
                    __m256 xydzr = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_ydzr_idx + i), c2);

                    __m256 xxlyuzf = _mm256_loadu_ps(inmap_ptr + inmap_xlyuzf_idx + i);
                    __m256 xxryuzf = _mm256_loadu_ps(inmap_ptr + inmap_xryuzf_idx + i);
                    __m256 xxlydzf = _mm256_loadu_ps(inmap_ptr + inmap_xlydzf_idx + i);
                    __m256 xxrydzf = _mm256_loadu_ps(inmap_ptr + inmap_xrydzf_idx + i);
                    __m256 xxlyuzr = _mm256_loadu_ps(inmap_ptr + inmap_xlyuzr_idx + i);
                    __m256 xxryuzr = _mm256_loadu_ps(inmap_ptr + inmap_xryuzr_idx + i);
                    __m256 xxlydzr = _mm256_loadu_ps(inmap_ptr + inmap_xlydzr_idx + i);
                    __m256 xxrydzr = _mm256_loadu_ps(inmap_ptr + inmap_xrydzr_idx + i);

                    __m256 yluf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyu, xxlyu)), _mm256_add_ps(_mm256_add_ps(xzf, xxlzf), _mm256_add_ps(xyuzf, xxlyuzf))));
                    __m256 yruf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyu, xxryu)), _mm256_add_ps(_mm256_add_ps(xzf, xxrzf), _mm256_add_ps(xyuzf, xxryuzf))));
                    __m256 yldf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyd, xxlyd)), _mm256_add_ps(_mm256_add_ps(xzf, xxlzf), _mm256_add_ps(xydzf, xxlydzf))));
                    __m256 yrdf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyd, xxryd)), _mm256_add_ps(_mm256_add_ps(xzf, xxrzf), _mm256_add_ps(xydzf, xxrydzf))));
                    __m256 ylur = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyu, xxlyu)), _mm256_add_ps(_mm256_add_ps(xzr, xxlzr), _mm256_add_ps(xyuzr, xxlyuzr))));
                    __m256 yrur = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyu, xxryu)), _mm256_add_ps(_mm256_add_ps(xzr, xxrzr), _mm256_add_ps(xyuzr, xxryuzr))));
                    __m256 yldr = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyd, xxlyd)), _mm256_add_ps(_mm256_add_ps(xzr, xxlzr), _mm256_add_ps(xydzr, xxlydzr))));
                    __m256 yrdr = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyd, xxryd)), _mm256_add_ps(_mm256_add_ps(xzr, xxrzr), _mm256_add_ps(xydzr, xxrydzr))));

                    _mm256_storeu_ps(outmap_ptr + outmap_idx + luf + i, yluf);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + ruf + i, yruf);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + ldf + i, yldf);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + rdf + i, yrdf);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + lur + i, ylur);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + rur + i, yrur);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + ldr + i, yldr);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + rdr + i, yrdr);
                }
                if (k > 0) {
                    __m256 xc = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_c_idx + j, mask), c8);

                    __m256 xxl = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xl_idx + j, mask), c4);
                    __m256 xxr = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xr_idx + j, mask), c4);
                    __m256 xyu = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_yu_idx + j, mask), c4);
                    __m256 xyd = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_yd_idx + j, mask), c4);
                    __m256 xzf = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_zf_idx + j, mask), c4);
                    __m256 xzr = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_zr_idx + j, mask), c4);

                    __m256 xxlyu = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xlyu_idx + j, mask), c2);
                    __m256 xxryu = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xryu_idx + j, mask), c2);
                    __m256 xxlyd = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xlyd_idx + j, mask), c2);
                    __m256 xxryd = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xryd_idx + j, mask), c2);
                    __m256 xxlzf = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xlzf_idx + j, mask), c2);
                    __m256 xxrzf = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xrzf_idx + j, mask), c2);
                    __m256 xxlzr = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xlzr_idx + j, mask), c2);
                    __m256 xxrzr = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_xrzr_idx + j, mask), c2);
                    __m256 xyuzf = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_yuzf_idx + j, mask), c2);
                    __m256 xydzf = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_ydzf_idx + j, mask), c2);
                    __m256 xyuzr = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_yuzr_idx + j, mask), c2);
                    __m256 xydzr = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_ydzr_idx + j, mask), c2);

                    __m256 xxlyuzf = _mm256_maskload_ps(inmap_ptr + inmap_xlyuzf_idx + j, mask);
                    __m256 xxryuzf = _mm256_maskload_ps(inmap_ptr + inmap_xryuzf_idx + j, mask);
                    __m256 xxlydzf = _mm256_maskload_ps(inmap_ptr + inmap_xlydzf_idx + j, mask);
                    __m256 xxrydzf = _mm256_maskload_ps(inmap_ptr + inmap_xrydzf_idx + j, mask);
                    __m256 xxlyuzr = _mm256_maskload_ps(inmap_ptr + inmap_xlyuzr_idx + j, mask);
                    __m256 xxryuzr = _mm256_maskload_ps(inmap_ptr + inmap_xryuzr_idx + j, mask);
                    __m256 xxlydzr = _mm256_maskload_ps(inmap_ptr + inmap_xlydzr_idx + j, mask);
                    __m256 xxrydzr = _mm256_maskload_ps(inmap_ptr + inmap_xrydzr_idx + j, mask);

                    __m256 yluf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyu, xxlyu)), _mm256_add_ps(_mm256_add_ps(xzf, xxlzf), _mm256_add_ps(xyuzf, xxlyuzf))));
                    __m256 yruf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyu, xxryu)), _mm256_add_ps(_mm256_add_ps(xzf, xxrzf), _mm256_add_ps(xyuzf, xxryuzf))));
                    __m256 yldf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyd, xxlyd)), _mm256_add_ps(_mm256_add_ps(xzf, xxlzf), _mm256_add_ps(xydzf, xxlydzf))));
                    __m256 yrdf = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyd, xxryd)), _mm256_add_ps(_mm256_add_ps(xzf, xxrzf), _mm256_add_ps(xydzf, xxrydzf))));
                    __m256 ylur = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyu, xxlyu)), _mm256_add_ps(_mm256_add_ps(xzr, xxlzr), _mm256_add_ps(xyuzr, xxlyuzr))));
                    __m256 yrur = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyu, xxryu)), _mm256_add_ps(_mm256_add_ps(xzr, xxrzr), _mm256_add_ps(xyuzr, xxryuzr))));
                    __m256 yldr = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxl), _mm256_add_ps(xyd, xxlyd)), _mm256_add_ps(_mm256_add_ps(xzr, xxlzr), _mm256_add_ps(xydzr, xxlydzr))));
                    __m256 yrdr = _mm256_mul_ps(cinv27, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(xc, xxr), _mm256_add_ps(xyd, xxryd)), _mm256_add_ps(_mm256_add_ps(xzr, xxrzr), _mm256_add_ps(xydzr, xxrydzr))));

                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + luf + j, mask, yluf);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + ruf + j, mask, yruf);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + ldf + j, mask, yldf);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + rdf + j, mask, yrdf);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + lur + j, mask, ylur);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + rur + j, mask, yrur);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + ldr + j, mask, yldr);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + rdr + j, mask, yrdr);
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Zoom::LinearZoom3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int th, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth * 2;
    unsigned int outheight = inheight * 2;
    unsigned int outdepth = indepth * 2;

    Util::CheckLength(channels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * outdepth * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    linear_zoom_3d(channels, inwidth, inheight, indepth, outwidth, outheight, outdepth, th, inmap_ptr, outmap_ptr);
}
