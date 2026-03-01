
#ifdef __cplusplus
using uint = unsigned int;
using float2 = TVector2<float>;
using float4x4 = TMatrix4x4<float>;

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
    Filter filter;
    int medium_id; // for participating media rendering in homework 2
};

#ifdef __cplusplus
}
#endif
