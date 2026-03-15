import Shared;
import Intersection;
import Frame;

import Lambertian;
import RoughPlastic;
import RoughDielectric;
import DisneyDiffuse;
import DisneyMetal;
import DisneyClearcoat;
import DisneyGlass;
import DisneySheen;
import DisneyBSDF;

float3 eval(in Material mat, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    switch (mat.type)
    {
    case MaterialType.Lambertian:      return eval_lambertian(mat, vertex, dir_in, dir_out);
    case MaterialType.RoughPlastic:    return eval_rough_plastic(mat, vertex, dir_in, dir_out);
    case MaterialType.RoughDielectric: return eval_rough_dielectric(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyDiffuse:   return eval_disney_diffuse(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyMetal:     return eval_disney_metal(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyClearcoat: return eval_disney_clearcoat(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyGlass:     return eval_disney_glass(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneySheen:     return eval_disney_sheen(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyBSDF:      return eval_disney_bsdf(mat, vertex, dir_in, dir_out);
    }
    return 0;
}

float pdf_sample_bsdf(in Material mat, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    switch (mat.type)
    {
    case MaterialType.Lambertian:      return pdf_lambertian(mat, vertex, dir_in, dir_out);
    case MaterialType.RoughPlastic:    return pdf_rough_plastic(mat, vertex, dir_in, dir_out);
    case MaterialType.RoughDielectric: return pdf_rough_dielectric(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyDiffuse:   return pdf_disney_diffuse(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyMetal:     return pdf_disney_metal(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyClearcoat: return pdf_disney_clearcoat(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyGlass:     return pdf_disney_glass(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneySheen:     return pdf_disney_sheen(mat, vertex, dir_in, dir_out);
    case MaterialType.DisneyBSDF:      return pdf_disney_bsdf(mat, vertex, dir_in, dir_out);
    }
    return 0;
}

Optional<BSDFSampleRecord> sample_bsdf(in Material mat, in PathVertex vertex, float3 dir_in, float3 rnd_param)
{
    switch (mat.type)
    {
    case MaterialType.Lambertian:      return sample_lambertian(mat, vertex, dir_in, rnd_param);
    case MaterialType.RoughPlastic:    return sample_rough_plastic(mat, vertex, dir_in, rnd_param);
    case MaterialType.RoughDielectric: return sample_rough_dielectric(mat, vertex, dir_in, rnd_param);
    case MaterialType.DisneyDiffuse:   return sample_disney_diffuse(mat, vertex, dir_in, rnd_param);
    case MaterialType.DisneyMetal:     return sample_disney_metal(mat, vertex, dir_in, rnd_param);
    case MaterialType.DisneyClearcoat: return sample_disney_clearcoat(mat, vertex, dir_in, rnd_param);
    case MaterialType.DisneyGlass:     return sample_disney_glass(mat, vertex, dir_in, rnd_param);
    case MaterialType.DisneySheen:     return sample_disney_sheen(mat, vertex, dir_in, rnd_param);
    case MaterialType.DisneyBSDF:      return sample_disney_bsdf(mat, vertex, dir_in, rnd_param);
    }
    return none;
}

// TODO: callable shaders instead of switch statements
#if 0
struct MaterialPayload {
    Material bsdf;
    PathVertex vertex;
    float3 dir_in;
    float pdf;
    float3 dir_out;
    float eta;
    float3 f_or_dir;
    float roughness;
};

[shader("callable")]
void eval_lambertian_bsdf(inout MaterialPayload payload)
{
}

[shader("callable")]
void sample_lambertian_bsdf(inout MaterialPayload payload)
{
}
#endif
