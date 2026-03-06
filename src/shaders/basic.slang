#include "camera.slang"

// ------------------------------------------------------------
// Global Resources
// ------------------------------------------------------------

[[vk::binding(0)]] RaytracingAccelerationStructure scene;
[[vk::binding(1)]] RWTexture2D<float4> image;
[[vk::binding(2)]] ConstantBuffer<Camera> camera;

// ------------------------------------------------------------
// Payload & Attributes
// ------------------------------------------------------------

struct Payload
{
    float3 throughput;
    float3 color;
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

    RayDesc ray = camera.sample_primary(float2(pixel_index) / float2(dimensions));

    Payload payload;
    payload.throughput = float3(1.0);
    payload.color = float3(0.0);
    // RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
    TraceRay(scene, 0, uint.maxValue, 0, 0, 0, ray, payload);
    image.Store(pixel_index, float4(payload.color, 1));
}

// ------------------------------------------------------------
// Miss Shader
// ------------------------------------------------------------

[shader("miss")]
void miss(inout Payload payload)
{
    payload.color = float3(0.1, 0.1, 0.3); // sky color
}

// ------------------------------------------------------------
// Closest Hit Shader
// ------------------------------------------------------------

static const float3 colors[] = {
	float3(1.0, 0.0, 0.0),
	float3(1.0, 1.0, 0.0),
	float3(0.0, 1.0, 0.0),
	float3(0.0, 1.0, 1.0),
	float3(0.0, 0.0, 1.0),
	float3(1.0, 1.0, 1.0),
};

[shader("closesthit")]
void chit(inout Payload payload, in HitAttributes attr)
{
#if 1
    float3 bary = float3(
		1.0 - attr.barycentrics.x - attr.barycentrics.y,
        attr.barycentrics.x,
        attr.barycentrics.y
    );
	
    // Simple barycentric color
    payload.color = bary;
    payload.hit   = 1;
#else
	payload.color = colors[InstanceID() % 6];
#endif
}
