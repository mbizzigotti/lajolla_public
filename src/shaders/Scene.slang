import Shared;
import Camera;

[[vk::push_constant]] ConstantBuffer<PerFrameInfo> frame;

struct Scene
{
    RWTexture2D<float4>              image;
	RaytracingAccelerationStructure  AS;
    ConstantBuffer<SceneInfo>        info;
    StructuredBuffer<Material>       materials;
    StructuredBuffer<Shape>          shapes;
    ByteAddressBuffer                positions;
    ByteAddressBuffer                triangles;
    ByteAddressBuffer                uvs;
    ByteAddressBuffer                normals;
	StructuredBuffer<Light>          lights;
	StructuredBuffer<TextureInfo>    texture_infos;
	ByteAddressBuffer                values; // For light sampling!
	StructuredBuffer<Medium>         media;
	SamplerState                     sampler;
	Texture2D                        textures[MAX_TEXTURE_COUNT];
	SamplerState                     volume_sampler;
	Texture3D<float>                 volumes[MAX_TEXTURE_COUNT];

    func accumulate_sample(uint2 pixel_index, float4 result)
    {
        if (any(isnan(result)))
        {
            result = 0; // Just throw out bad samples
          //  result = 1e9 * float4(1, 0, 1, 1);
		}
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
	
    property EnvironmentMap envmap     { get { return info.envmap; } }
    property Camera         camera     { get { return info.camera; } }
    property Options        options    { get { return info.options; } }
    property TableDist1D    light_dist { get { return info.light_dist; } }

    property bool has_envmap { get { return info.envmap.light_id != -1; } }

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

    RayDesc make_ray(float3 origin, float3 direction)
    {
        return {
		/* Origin    */ origin,
		/* TMin      */ intersection_epsilon(),
		/* Direction */ direction,
		/* TMax      */ float.maxValue
		};
	}
};

ParameterBlock<Scene> scene;
