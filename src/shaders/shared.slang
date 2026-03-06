
#ifdef __cplusplus
using uint = unsigned int;
using int2 = TVector2<int>;
using float2 = TVector2<float>;
using float3 = TVector3<float>;
using float4x4 = TMatrix4x4<float>;
#define interface struct

namespace GPU {
#endif

enum FilterType : uint
{
    Box,
    Tent,
    Gaussian,
};

struct Filter
{
    uint type;
    float value;
};

struct Camera
{
    float4x4 sample_to_cam, cam_to_sample;
    float4x4 cam_to_world, world_to_cam;
    int width, height;
    int2 _padding;
    Filter filter;
    int medium_id; // for participating media rendering in homework 2
};

struct Shape
{
	int material_id;
	int area_light_id;
    int interior_medium_id;
    int exterior_medium_id;
};

interface Material {};

enum class MaterialType {
    Lambertian,
    RoughPlastic,
    RoughDielectric,
    DisneyDiffuse,
    DisneyMetal,
    DisneyGlass,
    DisneyClearcoat,
    DisneySheen,
    DisneyBSDF,
};

struct Lambertian : Material
{
    float3 reflectance;
    float _padding;
};

struct RoughPlastic : Material
{
    float3 diffuse_reflectance;
    float3 specular_reflectance;
    float roughness;
    float eta; // internal IOR / externalIOR
};

struct RoughDielectric : Material
{
    float3 specular_reflectance;
    float3 specular_transmittance;
    float roughness;
    float eta; // internal IOR / externalIOR
};

struct DisneyDiffuse : Material
{
    float3 base_color;
    float roughness;
    float subsurface;
};

struct DisneyMetal : Material
{
    float3 base_color;
    float roughness;
    float anisotropic;
};

struct DisneyGlass : Material
{
    float3 base_color;
    float roughness;
    float anisotropic;
    float eta; // internal IOR / externalIOR
};

struct DisneyClearcoat : Material
{
    float clearcoat_gloss;
};

struct DisneySheen : Material
{
    float3 base_color;
    float sheen_tint;
};

struct DisneyBSDF : Material
{
    float3 base_color;
    float specular_transmission;
    float metallic;
    float subsurface;
    float specular;
    float roughness;
    float specular_tint;
    float anisotropic;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;
    float eta;
};

#ifdef __cplusplus
}
#endif
