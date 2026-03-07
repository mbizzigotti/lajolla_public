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
[[vk::binding(5)]] StructuredBuffer<float3, ScalarDataLayout> b_Positions;
[[vk::binding(6)]] StructuredBuffer<uint3, ScalarDataLayout> b_Triangles;
[[vk::binding(7)]] StructuredBuffer<float2> uv_buffer;
[[vk::binding(8)]] StructuredBuffer<float3> normal_buffer;

// ------------------------------------------------------------
// Payload & Attributes
// ------------------------------------------------------------

struct Payload
{
    pcg32_state rng;
    float3 throughput;
    float3 radiance;
    float3 dir;
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
    payload.rng = rng;
    payload.throughput = float3(1.0);
    payload.radiance = float3(0.0);
    {
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
// Any Hit Shader
// ------------------------------------------------------------

[shader("anyhit")]
void ahit(inout Payload payload, in HitAttributes attr)
{
    payload.hit = true;
}

// ------------------------------------------------------------
// Closest Hit Shader
// ------------------------------------------------------------

/*
struct BSDFSampleRecord
{
    float3 dir;
    float eta, roughness;
};

const Vector3 &dir_in;
const PathVertex &vertex;
const Vector2 &rnd_param_uv;
const Real &rnd_param_w;

Optional<BSDFSampleRecord> lambertian_sample_bsdf(Lambertian bsdf)
{
    // For Lambertian, we importance sample the cosine hemisphere domain.
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // Incoming direction is below the surface.
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    return BSDFSampleRecord {
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0), Real(1)};
}*/

struct Frame
{
    float3 x, y, n;
};

struct ShadingInfo
{
    float2 uv;          // UV coordinates for texture mapping
    Frame shading_frame; // the coordinate basis for shading
    float mean_curvature; // 0.5 * (dN/du + dN/dv)
    float inv_uv_size;
};

void coordinate_system(in float3 n, out float3 x, out float3 y) {
    if (n[2] < float(-1 + 1e-6)) {
        x = float3(0, -1, 0);
    	y = float3(-1, 0, 0);
    } else {
        float a = 1 / (1 + n[2]);
        float b = -n[0] * n[1] * a;
        x = float3(1 - n[0] * n[0] * a, b, -n[0]);
		y = float3(b, 1 - n[1] * n[1] * a, -n[1]);
    }
}

vector<float, N> interpolate<int N>(vector<float, N> a, vector<float, N> b, vector<float, N> c, float2 bary)
{
    return a * bary.x + b * bary.y + c * (1 - bary.x - bary.y);
}

ShadingInfo triangle_mesh_shading_info(Shape shape, float2 bary)
{
    // Get UVs of the three vertices
    uint primitive_index = PrimitiveIndex() + shape.index_offset;
    uint3 index = b_Triangles.Load(primitive_index) + shape.vertex_offset;
    float2 uvs[3];
    if (shape.has_uvs == 1) {
        uvs[0] = uv_buffer[index[0]];
        uvs[1] = uv_buffer[index[1]];
        uvs[2] = uv_buffer[index[2]];
    } else {
        // Use barycentric coordinates
        uvs[0] = float2(0, 0);
        uvs[1] = float2(1, 0);
        uvs[2] = float2(1, 1);
    }
    float2 uv = interpolate(uvs[0], uvs[1], uvs[2], bary);
    float3 p0 = b_Positions.Load(index[0]),
           p1 = b_Positions.Load(index[1]),
           p2 = b_Positions.Load(index[2]);

    float3 geometric_normal = normalize(cross(p1 - p0, p2 - p0));

    // We want to derive dp/du & dp/dv. We have the following
    // relation:
    // p  = (1 - s - t) * p0   + s * p1   + t * p2
    // uv = (1 - s - t) * uvs0 + s * uvs1 + t * uvs2
    // dp/duv = dp/dst * dst/duv = dp/dst * (duv/dst)^{-1}
    // where dp/dst is a 3x2 matrix, duv/dst and dst/duv is a 2x2 matrix,
    // and dp/duv is a 3x2 matrix.

    // Let's build duv/dst first. To be clearer, it is
    // [du/ds, du/dt]
    // [dv/ds, dv/dt]
    float2 duvds = uvs[2] - uvs[0];
    float2 duvdt = uvs[2] - uvs[1];
    // The inverse of this matrix is
    // (1/det) [ dv/dt, -du/dt]
    //         [-dv/ds,  du/ds]
    // where det = duds * dvdt - dudt * dvds
    float det = duvds[0] * duvdt[1] - duvdt[0] * duvds[1];
    float dsdu = duvdt[1] / det;
    float dtdu = -duvds[1] / det;
    float dsdv = duvdt[0] / det;
    float dtdv = -duvds[0] / det;
    float3 dpdu, dpdv;
    if (fabs(det) > 1e-8f) {
        // Now we just need to do the matrix multiplication
        float3 dpds = p2 - p0;
        float3 dpdt = p2 - p1;
        dpdu = dpds * dsdu + dpdt * dtdu;
        dpdv = dpds * dsdv + dpdt * dtdv;
    } else {
        // degenerate uvs. Use an arbitrary coordinate system
        coordinate_system(geometric_normal, dpdu, dpdv);
    }

    // Now let's get the shading normal & mean_curvature.
    // By default it is the geometry normal and we have zero curvature.
    float3 shading_normal = geometric_normal;
    float mean_curvature = 0;
    float3 tangent, bitangent;
    // However if we have vertex normals, that overrides the geometry normal.
    if (shape.has_normals == 1) {
        float3 n0 = normal_buffer[index[0]],
               n1 = normal_buffer[index[1]],
               n2 = normal_buffer[index[2]];
        shading_normal = normalize(
            (1 - bary[0] - bary[1]) * n0 +
            bary[0] * n1 +
            bary[1] * n2);
        // dpdu may not be orthogonal to shading normal:
        // subtract the projection of shading_normal onto dpdu to make them orthogonal
        tangent = normalize(dpdu - shading_normal * dot(shading_normal, dpdu));

        // We want to compute dn/du & dn/dv for mean curvature.
        // This is computed in a similar way to dpdu.
        // dn/duv = dn/dst * dst/duv = dn/dst * (duv/dst)^{-1}
        float3 dnds = n2 - n0;
        float3 dndt = n2 - n1;
        float3 dndu = dnds * dsdu + dndt * dtdu;
        float3 dndv = dnds * dsdv + dndt * dtdv;
        bitangent = normalize(cross(shading_normal, tangent));
        mean_curvature = (dot(dndu, tangent) +
                          dot(dndv, bitangent)) / 2;
    } else {
        tangent = normalize(dpdu - shading_normal * dot(shading_normal, dpdu));
        bitangent = normalize(cross(shading_normal, tangent));
    }

    Frame shading_frame = { tangent, bitangent, shading_normal };
    return { uv, shading_frame, mean_curvature, max(length(dpdu), length(dpdv)) };
}

[shader("closesthit")]
void chit(inout Payload payload, in HitAttributes attr)
{
    uint instance_id = InstanceID();
    Shape shape = shape_buffer[instance_id];
    // float3 reflectance = material_buffer[shape.material_id].reflectance;
    // payload.throughput *= reflectance;
    // payload.radiance =
    ShadingInfo shading_info = triangle_mesh_shading_info(shape, attr.barycentrics);
    payload.radiance = shading_info.shading_frame.n;
    payload.hit = 1;
}
