
#ifdef __cplusplus
using uint = unsigned int;
using int2 = TVector2<int>;
using float2 = TVector2<float>;
using float3 = TVector3<float>;
using float4x4 = TMatrix4x4<float>;
#define interface struct

namespace GPU {
#endif

struct PerFrameInfo
{
	uint index;
};

struct BSphere
{
    float3 center;
    float  radius;
};

struct Options
{
    BSphere bounds;
    int max_depth;
    int rr_depth;
    int max_null_collisions;
    int envmap_light_id;
};

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

static const int SHAPE_HAS_UVS      = 0x01;
static const int SHAPE_HAS_NORMALS  = 0x02;
static const int SHAPE_IS_SPHERE    = 0x04;

struct Shape
{
	int material_id;
	int area_light_id;
    int interior_medium_id;
    int exterior_medium_id;
    uint vertex_offset;
    uint index_offset;
    int flags;
    int _padding;
	// Only if is a sphere!!
    float3 position;
    float radius;
};

struct FloatArray
{
    uint offset;
    uint count;
};

struct TableDist1D
{
    FloatArray pmf, cdf;
};

struct TableDist2D
{
    FloatArray cdf_rows, pdf_rows;
    // cdf_maringlas & pdf_marginals store a single 1D piecewise
    // constant distribution for sampling a row
    FloatArray cdf_marginals, pdf_marginals;
    float total_values;
    int width, height;
};

struct TextureInfo
{
	float uscale, vscale;
    float uoffset, voffset;
};

/// An environment map (Envmap) is an infinitely far area light source
/// that covers the whole bounding spherical domain of the scene.
/// A texture is used to represent light coming from each direction.
struct EnvironmentMap
{
    float4x4 to_world, to_local;
    TableDist2D sampling_dist;
    float scale;
    int texture;
};

struct Light
{
    float3 intensity;
    int shape_id; // -1 if Envmap!
    float surface_area;
    TableDist1D triangle_sampler; // Only for triangle meshes!
};

struct SceneInfo
{
    Options options;
    Camera camera;
    EnvironmentMap envmap;
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
