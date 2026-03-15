import Shared;
import Frame;
import Scene;
import Ray;
import Light;

struct BSDFSampleRecord
{
    float3 dir_out;
    float eta;       // The index of refraction ratio. Set to 0 if it's not a transmission event.
    float roughness; // Roughness of the selected BRDF layer ([0, 1]).
};

struct PathVertex
{
    float3 position;
    float3 geometric_normal; // always face at the same direction at shading_frame.n
    Frame shading_frame;
    float2 st; // A 2D parametrization of the surface. Irrelavant to UV mapping.
                // for triangle this is the barycentric coordinates, which we use
                // for interpolating the uv map.
    float2 uv; // The actual UV we use for texture fetching.
    // For texture filtering, stores approximatedly min(abs(du/dx), abs(dv/dx), abs(du/dy), abs(dv/dy))
    float uv_screen_size;
    float mean_curvature; // For ray differential propagation.
    float ray_radius; // For ray differential propagation.
    int shape_id = -1;
    int primitive_id = -1; // For triangle meshes. This indicates which triangle it hits.
    int material_id = -1;
    int area_light_id = -1;

    // If the path vertex is inside a medium, these two IDs
    // are the same.
    int interior_medium_id = -1;
    int exterior_medium_id = -1;
};

struct ShadingInfo
{
    Frame shading_frame;  // the coordinate basis for shading
    float2 uv;            // UV coordinates for texture mapping
    float mean_curvature; // 0.5 * (dN/du + dN/dv)
    float inv_uv_size;
};

struct HitPayload
{
    ShadingInfo shading_info;
    float3 geometric_normal;
    float t_hit;
    float2 st;
    uint shape_id;
	int material_id;
	int area_light_id;
    int interior_medium_id;
    int exterior_medium_id;
    bool hit;
};

struct SphereHitAttributes
{
    float3 normal;
    float2 uv;
};

vector<float, N> interpolate<int N>(vector<float, N> a, vector<float, N> b, vector<float, N> c, float2 bary)
{
    return a * (1 - bary.x - bary.y) + b * bary.x + c * bary.y;
}

// ------------------------------------------------------------
// Closest Hit Shader (Triangle Mesh)
// ------------------------------------------------------------

[shader("closesthit")]
void chit_triangle(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    uint instance_id = InstanceID();
    uint instance_index = InstanceIndex();
    uint primitive_index = PrimitiveIndex();

    payload.shape_id = instance_id;
    payload.st = attr.barycentrics;

    Shape shape = scene.shapes.Load(instance_id);

    payload.material_id = shape.material_id;
    payload.area_light_id = shape.area_light_id;
    payload.interior_medium_id = shape.interior_medium_id;
    payload.exterior_medium_id = shape.exterior_medium_id;

    // Get UVs of the three vertices
    uint3 index = scene.triangles.LoadByteOffset<uint3>((primitive_index + shape.index_offset) * sizeof(uint3)) + shape.vertex_offset;

    float2 uvs[3];
    if (bool(shape.flags & SHAPE_HAS_UVS)) {
        uvs[0] = scene.uvs.LoadByteOffset<float2>((index[0]) * sizeof(float2));
        uvs[1] = scene.uvs.LoadByteOffset<float2>((index[1]) * sizeof(float2));
        uvs[2] = scene.uvs.LoadByteOffset<float2>((index[2]) * sizeof(float2));
    } else {
        // Use barycentric coordinates
        uvs[0] = float2(0, 0);
        uvs[1] = float2(1, 0);
        uvs[2] = float2(1, 1);
    }
    float2 uv = interpolate(uvs[0], uvs[1], uvs[2], attr.barycentrics);
    float3 p0 = scene.positions.LoadByteOffset<float3>((index[0]) * sizeof(float3)),
           p1 = scene.positions.LoadByteOffset<float3>((index[1]) * sizeof(float3)),
           p2 = scene.positions.LoadByteOffset<float3>((index[2]) * sizeof(float3));

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
    if (bool(shape.flags & SHAPE_HAS_NORMALS)) {
        float3 n0 = scene.normals.LoadByteOffset<float3>((index[0]) * sizeof(float3)),
               n1 = scene.normals.LoadByteOffset<float3>((index[1]) * sizeof(float3)),
               n2 = scene.normals.LoadByteOffset<float3>((index[2]) * sizeof(float3));
        shading_normal = interpolate(n0, n1, n2, attr.barycentrics);
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
    payload.geometric_normal = geometric_normal;
    payload.shading_info = { shading_frame, uv, mean_curvature, max(length(dpdu), length(dpdv)) };
    payload.t_hit = RayTCurrent();
    payload.hit = true;
}

// ------------------------------------------------------------
// Closest Hit Shader (Sphere)
// ------------------------------------------------------------

[shader("closesthit")]
void chit_sphere(inout HitPayload payload, in SphereHitAttributes attr)
{
    uint instance_id = InstanceID();
    payload.shape_id = instance_id;
    payload.st = attr.uv;
    
	Shape shape = scene.shapes.Load(instance_id);
    payload.material_id = shape.material_id;
    payload.area_light_id = shape.area_light_id;
    payload.interior_medium_id = shape.interior_medium_id;
    payload.exterior_medium_id = shape.exterior_medium_id;
	
    // To compute the shading frame, we use the geometry normal as normal,
    // and dpdu as one of the tangent vector. 
    // We use the azimuthal angle as u, and the elevation as v, 
    // thus the point p on sphere and u, v has the following relationship:
    // p = center + {r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v)}
    // thus dpdu = {-r * sin(u) * sin(v), r * cos(u) * sin(v), 0}
    //      dpdv = { r * cos(u) * cos(v), r * sin(u) * cos(v), - r * sin(v)}
    float3 dpdu = {-shape.radius * sin(attr.uv[0]) * sin(attr.uv[1]),
                    shape.radius * cos(attr.uv[0]) * sin(attr.uv[1]),
                    0};
    float3 dpdv = { shape.radius * cos(attr.uv[0]) * cos(attr.uv[1]),
                    shape.radius * sin(attr.uv[0]) * cos(attr.uv[1]),
                   -shape.radius * sin(attr.uv[1])};
    // dpdu may not be orthogonal to shading normal:
    // subtract the projection of shading_normal onto dpdu to make them orthogonal
    float3 tangent = normalize(dpdu - attr.normal * dot(attr.normal, dpdu));
    Frame shading_frame = Frame(tangent, normalize(cross(attr.normal, tangent)), attr.normal);
	
    payload.geometric_normal = attr.normal;
    payload.shading_info = ShadingInfo(shading_frame, attr.uv,
						1 / shape.radius, /* mean curvature */
                       (length(dpdu) + length(dpdv)) / 2);
    payload.t_hit = RayTCurrent();
    payload.hit = true;
}

/// Numerically stable quadratic equation solver at^2 + bt + c = 0
/// See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
/// returns false when it can't find solutions.
bool solve_quadratic(float a, float b, float c, out float t0, out float t1) {
    t0 = 0;
    t1 = 0;
    // Degenerated case
    if (a == 0) {
        if (b == 0) {
            return false;
        }
        t0 = -c / b;
		t1 = -c / b;
        return true;
    }

    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    }
    float root_discriminant = sqrt(discriminant);
    if (b >= 0) {
        t0 = (-b - root_discriminant) / (2 * a);
        t1 = 2 * c / (-b - root_discriminant);
    } else {
        t0 = 2 * c / (-b + root_discriminant);
        t1 = (-b + root_discriminant) / (2 * a);
    }
    return true;
}

[shader("intersection")]
void ints_sphere()
{
    uint instance_id = InstanceID();
    Shape sphere = scene.shapes.Load(instance_id);

    float3 rayOrigin = WorldRayOrigin();
    float3 rayDir = WorldRayDirection();

    // Our sphere is ||p - x||^2 = r^2
    // substitute x = o + d * t, we want to solve for t
    // ||p - (o + d * t)||^2 = r^2
    // (p.x - (o.x + d.x * t))^2 + (p.y - (o.y + d.y * t))^2 + (p.z - (o.z + d.z * t))^2 - r^2 = 0
    // (d.x^2 + d.y^2 + d.z^2) t^2 + 2 * (d.x * (o.x - p.x) + d.y * (o.y - p.y) + d.z * (o.z - p.z)) t +
    // ((p.x-o.x)^2 + (p.y-o.y)^2 + (p.z-o.z)^2  - r^2) = 0
    // A t^2 + B t + C
    float3 v = rayOrigin - sphere.position;
    float A = dot(rayDir, rayDir);
    float B = 2 * dot(rayDir, v);
    float C = dot(v, v) - sphere.radius * sphere.radius;
    float t0, t1;
    if (!solve_quadratic(A, B, C, t0, t1)) {
        // No intersection
        return;
    }
    // This can happen due to numerical inaccuracies
    if (t0 > t1) {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    float tnear = RayTMin();
    float tfar = RayTCurrent();

    float t = -1;
    if (t0 >= tnear && t0 < tfar) {
        t = t0;
    }
    if (t1 >= tnear && t1 < tfar && t < 0) {
        t = t1;
    }

    if (t >= tnear && t < tfar) {
        // Record the intersection
        SphereHitAttributes attr;
        float3 p = rayOrigin + t * rayDir;
        float3 geometric_normal = p - sphere.position;
        // rtc_hit->Ng doesn't need to be normalized
        attr.normal = {
            geometric_normal.x,
            geometric_normal.y,
            geometric_normal.z,
        };
        // We use the spherical coordinates as uv
        float3 cartesian = geometric_normal / sphere.radius;
        // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        // We use the convention that y is up axis.
        float elevation = acos(clamp(cartesian.y, -1, 1));
        float azimuth = atan2(cartesian.z, cartesian.x);
        attr.uv = {
            azimuth / (2 * float.getPi()),
            elevation / float.getPi(),
        };
		ReportHit(t, 0, attr);
    }
}

/// Intersect a ray with a scene. If the ray doesn't hit anything,
/// returns an invalid optional output.
Optional<PathVertex> intersect(RayDesc ray, RayDifferential ray_diff = { 0, 0 })
{
    HitPayload payload;
    payload.hit = false;
    TraceRay(scene.AS, 0, uint.maxValue, 0, 1, 0, ray, payload);

    if (!payload.hit) {
        return none;
	}

    PathVertex vertex;
    vertex.position = ray.Origin + ray.Direction * payload.t_hit;
    vertex.geometric_normal = payload.geometric_normal;
    vertex.shape_id = payload.shape_id;
    // vertex.primitive_id = payload.instance_index;
    vertex.area_light_id = payload.area_light_id;
    vertex.material_id = payload.material_id;
    vertex.interior_medium_id = payload.interior_medium_id;
    vertex.exterior_medium_id = payload.exterior_medium_id;
    vertex.st = payload.st;
    vertex.shading_frame = payload.shading_info.shading_frame;
    vertex.uv = payload.shading_info.uv;
    vertex.mean_curvature = payload.shading_info.mean_curvature;
    vertex.ray_radius = transfer(ray_diff, distance(ray.Origin, vertex.position));
    // vertex.ray_radius stores approximatedly dp/dx,
    // we get uv_screen_size (du/dx) using (dp/dx)/(dp/du)
    vertex.uv_screen_size = vertex.ray_radius / payload.shading_info.inv_uv_size;

    // Flip the geometry normal to the same direction as the shading normal
    if (dot(vertex.geometric_normal, vertex.shading_frame.n) < 0) {
        vertex.geometric_normal = -vertex.geometric_normal;
    }

    return vertex;
}

struct ShadowPayload
{
    bool hit;
};

// ------------------------------------------------------------
// Miss Shader
// ------------------------------------------------------------

[shader("miss")]
void miss(inout HitPayload payload)
{
    payload.hit = false;
}

// ------------------------------------------------------------
// Miss Shader (Shadow)
// ------------------------------------------------------------

[shader("miss")]
void miss_shadow(inout ShadowPayload payload)
{
    payload.hit = false;
}

/// Test is a ray segment intersect with anything in a scene.
bool occluded(RayDesc ray)
{
    ShadowPayload payload;
    payload.hit = true;
    int flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
	          | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
    TraceRay(scene.AS, flags, uint.maxValue, 0, 0, 1, ray, payload);
    return payload.hit;
}

/// Computes the emission at a path vertex v, with the viewing direction
/// pointing outwards of the intersection.
float3 emission(in PathVertex v, float3 view_dir)
{
    uint light_id = v.area_light_id;
    Light light = scene.lights[light_id];
    return emission(light, view_dir, v.uv_screen_size, { v.position, v.geometric_normal });
}
