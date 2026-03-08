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
	
    property EnvironmentMap envmap  { get { return info.envmap; } }
    property Camera         camera  { get { return info.camera; } }
    property Options        options { get { return info.options; } }
};

ParameterBlock<Scene> scene;
