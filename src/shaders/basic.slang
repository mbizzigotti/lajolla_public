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

struct RayPayload
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

    image.Store(pixelIndex, float4(uv, 0.0, 1.0));
}

// ------------------------------------------------------------
// Miss Shader
// ------------------------------------------------------------

[shader("miss")]
void main(inout RayPayload payload)
{
    payload.color = float3(0.1, 0.1, 0.3); // sky color
}

// ------------------------------------------------------------
// Closest Hit Shader
// ------------------------------------------------------------

[shader("closesthit")]
void main(inout RayPayload payload, in HitAttributes attr)
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
