
#ifdef __cplusplus
using uint = unsigned int;
using int2 = TVector2<int>;
using int3 = TVector3<int>;
using int4 = TVector4<int>;
using uint4 = TVector4<uint32_t>;
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
    int max_depth;
    int rr_depth;
    int max_null_collisions;
    int _padding;
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
    int2 _padding;
};

struct Camera
{
    float4x4 sample_to_cam, cam_to_sample;
    float4x4 cam_to_world, world_to_cam;
    int width, height;
    int medium_id; // for participating media rendering in homework 2
    int _padding;
    Filter filter;
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

struct TableDist1D
{
    uint4 _pmf_cdf;
};

struct TableDist2D
{
    uint4 _pdf_cdf_rows;
    uint4 _pdf_cdf_marginals;
    float total_values;
    int width, height;
    int _padding;
};

static const int MAX_TEXTURE_COUNT = 16;

struct TextureInfo
{
    float3 color0;
    int index;
	float3 color1;
    int is_checkerboard;
    float2 scale;
    float2 offset;
};

struct float3_parameter
{
    float3 value;
    int texture_id;
};

struct float_parameter
{
    float3 value;
    int texture_id;
};

struct EnvironmentMap
{
    float4x4 to_world;
	float4x4 to_local;
    TableDist2D sampling_dist;
    float3_parameter values;
    int2 _padding;
    float scale;
    int light_id;
};

struct Light
{
    TableDist1D triangle_sampler; // Only for triangle meshes!
    float3 intensity;
    float surface_area;
    int3 _padding;
    int shape_id; // -1 if Envmap!
};

struct SceneInfo
{
	Options options;
    Camera camera;
    TableDist1D light_dist;
	EnvironmentMap envmap;
    BSphere bounds;
};

interface Material {};

enum class MaterialType
{
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
    float3_parameter reflectance;
};

struct RoughPlastic : Material
{
    float3 diffuse_reflectance;
    float roughness;
    float3 specular_reflectance;
    float eta; // internal IOR / externalIOR
};

struct RoughDielectric : Material
{
    float3 specular_reflectance;
    float roughness;
    float3 specular_transmittance;
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
