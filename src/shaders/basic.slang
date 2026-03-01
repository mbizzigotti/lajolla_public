// ------------------------------------------------------------
// Global Resources
// ------------------------------------------------------------

[[vk::binding(0)]] RWTexture2D<float4> image;
[[vk::binding(1)]] RaytracingAccelerationStructure scene;

#if 0
// Camera parameters
[[vk::binding(2)]] cbuffer CameraData
{
    float4x4 viewInverse;
    float4x4 projInverse;
    float3   cameraPosition;
    float    padding;
};
#endif

// ------------------------------------------------------------
// Payload & Attributes
// ------------------------------------------------------------

struct Payload
{
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
void main()
{
    uint2 pixelIndex = DispatchRaysIndex().xy;
    uint2 dims       = DispatchRaysDimensions().xy;

    float2 uv = float2(pixelIndex) / float2(dims);

    RayDesc rayDesc;
    rayDesc.Origin = float3(0, 0, 0);
    rayDesc.Direction = normalize(float3(uv, 3));
    rayDesc.TMin = 0.001;
    rayDesc.TMax = 10000.0;

    Payload payload;
    payload.color = float3(uv, 0.0);
	
    TraceRay(scene, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, rayDesc, payload);

    image.Store(pixelIndex, float4(payload.color, 1.0));
}

// ------------------------------------------------------------
// Miss Shader
// ------------------------------------------------------------

[shader("miss")]
void main(inout Payload payload)
{
    payload.color = float3(0.1, 0.1, 0.3); // sky color
}

// ------------------------------------------------------------
// Closest Hit Shader
// ------------------------------------------------------------

[shader("closesthit")]
void main(inout Payload payload, in HitAttributes attr)
{
    float3 bary = float3(
        1.0 - attr.barycentrics.x - attr.barycentrics.y,
        attr.barycentrics.x,
        attr.barycentrics.y
    );

    // Simple barycentric color
    payload.color = bary;
    payload.hit   = 1;
}
