import shared;
import camera;
import random;

// ------------------------------------------------------------
// Global Resources
// ------------------------------------------------------------

[[vk::push_constant]] ConstantBuffer<uint4> frame;
[[vk::binding(0)]] RaytracingAccelerationStructure scene;
[[vk::binding(1)]] RWTexture2D<float4> image;
[[vk::binding(2)]] ConstantBuffer<Camera> camera;
[[vk::binding(3)]] StructuredBuffer<Lambertian> material_buffer;
[[vk::binding(4)]] StructuredBuffer<Shape> shape_buffer;

// ------------------------------------------------------------
// Payload & Attributes
// ------------------------------------------------------------

struct Payload
{
    float3 throughput;
    float3 radiance;
    uint   hit;
};

struct HitAttributes
{
    float2 barycentrics;
};

// ------------------------------------------------------------
// Ray Generation Shader
// ------------------------------------------------------------

[shader("raygeneration")]
void rgen()
{
    uint2 pixel_index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    pcg32_state rng = init_pcg32(frame.x * (pixel_index.y * dimensions.x + pixel_index.x));
    RayDesc ray = camera.sample_primary(pixel_index, dimensions, rng);

    Payload payload;
    payload.throughput = float3(1.0);
    payload.radiance = float3(0.0);
    {
		// RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
		TraceRay(scene, 0, uint.maxValue, 0, 0, 0, ray, payload);
    }
    float4 result = float4(payload.radiance, 1);
	
    if (frame.x == 0)
	{
		image.Store(pixel_index, result);
	}
    else
    {
		float u = 1.0f / (float)(frame.x + 1);
        float4 old = image.Load(pixel_index);
        image.Store(pixel_index, lerp(old, result, u));
    }
}

// ------------------------------------------------------------
// Miss Shader
// ------------------------------------------------------------

[shader("miss")]
void miss(inout Payload payload)
{
    payload.radiance = float3(0.1, 0.1, 0.3); // sky color
}

// ------------------------------------------------------------
// Closest Hit Shader
// ------------------------------------------------------------

[shader("closesthit")]
void chit(inout Payload payload, in HitAttributes attr)
{
#if 0
    float3 bary = float3(
		1.0 - attr.barycentrics.x - attr.barycentrics.y,
        attr.barycentrics.x,
        attr.barycentrics.y
    );
	
    // Simple barycentric color
    payload.radiance = bary;
    payload.hit   = 1;
#else
    uint instance_id = InstanceID();
    Shape shape = shape_buffer[instance_id];
    float3 reflectance = material_buffer[shape.material_id].reflectance;
    payload.radiance = reflectance;
#endif
}
