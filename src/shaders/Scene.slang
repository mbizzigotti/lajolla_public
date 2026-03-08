import shared;
import Camera;

[[vk::push_constant]] ConstantBuffer<PerFrameInfo> frame;

struct Scene
{
    RWTexture2D<float4>              image;
	RaytracingAccelerationStructure  AS;
    ConstantBuffer<SceneInfo>        info;
    StructuredBuffer<Lambertian>     materials;
    StructuredBuffer<Shape>          shapes;
    ByteAddressBuffer                positions;
    ByteAddressBuffer                triangles;
    StructuredBuffer<float2>         uvs;
    StructuredBuffer<float3>         normals;
	StructuredBuffer<Light>          lights;
	StructuredBuffer<TextureInfo>    texture_infos;
	StructuredBuffer<float>          values; // For light sampling!

    func accumulate_sample(uint2 pixel_index, float4 result)
    {
        if (frame.index == 0)
        {
            image.Store(pixel_index, result);
        }
        else
        {
            float u = 1.0f / (float)(frame.index + 1);
            float4 old = image.Load(pixel_index);
            image.Store(pixel_index, lerp(old, result, u));
        }
	}
	
    property EnvironmentMap envmap     { get { return {}; } }
    property Camera         camera     { get { return info.camera; } }
    property Options        options    { get { return info.options; } }
    property TableDist1D    light_dist { get { return info.light_dist; } }

    [ForceInline] float shadow_epsilon()       { return min(info.bounds.radius * 1e-5, 0.01); }
	[ForceInline] float intersection_epsilon() { return min(info.bounds.radius * 1e-5, 0.01); }

    RayDesc make_shadow_ray(float3 origin, float3 point)
    {
        float3 dir_light = point - origin;
        float d = length(dir_light);
        return {
		/* Origin    */ origin + shadow_epsilon() * (dir_light / d),
		/* TMin      */ 0,
		/* Direction */ dir_light / d,
		/* TMax      */ (1 - shadow_epsilon()) * d
        };
	}
};

ParameterBlock<Scene> scene;
