import shared;
import Utils;
import Scene;
import TableDist;
import Texture;

struct PointAndNormal
{
    float3 position;
    float3 normal;
};

[ForceInline]
bool is_envmap(Light light) {
    return light.shape_id == -1;
}

int sample_light(float u)
{
    return sample(scene.light_dist, u);
}

float light_pmf(int light_id)
{
    return pmf(scene.light_dist, light_id);
}

PointAndNormal sample_point_on_envmap(float3 rnd_param)
{
    float2 uv = sample(scene.envmap.sampling_dist, rnd_param.xy);
    // Convert uv to spherical coordinates
    float azimuth = uv[0] * (2 * float.getPi());
    float elevation = uv[1] * float.getPi();
    // Convert spherical coordinates to Cartesian coordinates.
    // (https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates)
    // We use the convention that y is the up axis.
    float3 local_dir = { sin(azimuth) * sin(elevation),
                         cos(elevation),
                        -cos(azimuth) * sin(elevation) };
    float3 world_dir = xform_vector(scene.envmap.to_world, local_dir);
    return { float3(0), -world_dir };
}

float pdf_point_on_envmap(in PointAndNormal point_on_light)
{
    // We store the direction pointing outwards from light in point_on_light.normal.
    float3 world_dir = -point_on_light.normal;
    // Convert the direction to local Catesian coordinates.
    float3 local_dir = xform_vector(scene.envmap.to_local, world_dir);
    // Convert the Cartesian coordinates to the spherical coordinates.
    // We use the convention that y is the up-axis.
    float2 uv = { atan2(local_dir[0], -local_dir[2]) * (0.5 / float.getPi()),
                  acos(clamp(local_dir[1], float(-1), float(1))) * (1 / float.getPi()) };
    // atan2 returns -pi to pi, we map [-pi, 0] to [pi, 2pi]
    if (uv[0] < 0) {
        uv[0] += 1;
    }
    float cos_elevation = local_dir.y;
    float sin_elevation = sqrt(saturate(1 - cos_elevation * cos_elevation));
    if (sin_elevation <= 0) {
        // degenerate
        return 0;
    }
    return pdf(scene.envmap.sampling_dist, uv) / (2 * float.getPi() * float.getPi() * sin_elevation);
}

float3 emission_envmap(float3 view_dir, float view_footprint)
{
    // View dir is pointing outwards "from" the light.
    // An environment map stores the light from the opposite direction,
    // so we need to flip view dir.
    // We then transform the direction to the local Cartesian coordinates.
    float3 local_dir = xform_vector(scene.envmap.to_local, -view_dir);
    // Convert the Cartesian coordinates to the spherical coordinates.
    float2 uv = { atan2(local_dir[0], -local_dir[2]) * (0.5 / float.getPi()),
                  acos(clamp(local_dir[1], -1, 1)) * (1 / float.getPi()) };
    // atan2 returns -pi to pi, we map [-pi, 0] to [pi, 2pi]
    if (uv[0] < 0) {
        uv[0] += 1;
    }

    // For envmap, view_footprint stores (approximatedly) d view_dir / dx
    // We want to convert it to du/dx -- we do it by computing (d dir / dx) * (d u / d dir)
    // To do this we differentiate through the process above:

    // We abbrevite local_dir as w
    float3 w = local_dir;
    float dudwx = -w.z / (w.x * w.x + w.z * w.z);
    float dudwz = w.x / (w.x * w.x + w.z * w.z);
    float dvdwy = -1 / sqrt(max(1 - w.y * w.y, 0));
    // We only want to know the length of dudw & dvdw
    // The local coordinate transformation is length preserving,
    // so we don't need to differentiate through it.
    float footprint = min(sqrt(dudwx * dudwx + dudwz * dudwz), dvdwy);

    return eval(scene.envmap.values, uv, footprint) * scene.envmap.scale;
}

PointAndNormal sample_point_on_triangle_mesh(in TableDist1D triangle_sampler, in Shape shape, float3 ref_point, float3 rnd_param)
{
    int tri_id = sample(triangle_sampler, rnd_param.z);
    uint3 index = scene.triangles.LoadByteOffset<uint3>((tri_id + shape.index_offset) * sizeof(uint3));
    float3 v0 = scene.positions.LoadByteOffset<float3>((index[0] + shape.vertex_offset) * sizeof(float3));
    float3 v1 = scene.positions.LoadByteOffset<float3>((index[1] + shape.vertex_offset) * sizeof(float3));
    float3 v2 = scene.positions.LoadByteOffset<float3>((index[2] + shape.vertex_offset) * sizeof(float3));
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    // https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#SamplingaTriangle
    float a = sqrt(saturate(rnd_param.x));
    float b1 = 1 - a;
    float b2 = a * rnd_param.y;
    float3 geometric_normal = normalize(cross(e1, e2));
    // Flip the geometric normal to the same side as the shading normal
    if (bool(shape.flags & SHAPE_HAS_NORMALS))
	{
        float3 n0 = scene.normals.LoadByteOffset<float3>((index[0] + shape.vertex_offset) * sizeof(float3));
        float3 n1 = scene.normals.LoadByteOffset<float3>((index[1] + shape.vertex_offset) * sizeof(float3));
        float3 n2 = scene.normals.LoadByteOffset<float3>((index[2] + shape.vertex_offset) * sizeof(float3));
        float3 shading_normal = normalize(
            (1 - b1 - b2) * n0 + b1 * n1 + b2 * n2);
        if (dot(geometric_normal, shading_normal) < 0) {
            geometric_normal = -geometric_normal;
        }
    }
    return { v0 + (e1 * b1) + (e2 * b2), geometric_normal };
}

/// Given some random numbers and a reference point, sample a point on the light source.
/// If the point is on a surface, returns both the point & normal on it.
/// If the point is infinitely far away (e.g., on an environment map),
/// we store the direction that points towards the origin in PointAndNormal.normal.
/// rnd_param_w is usually used for choosing a discrete element e.g., choosing a triangle in a mesh light.
/// rnd_param_uv is usually used for picking a point on that element.
PointAndNormal sample_point_on_light(in Light light, float3 ref_point, float3 rnd_param)
{
    if (is_envmap(light)) {
        return sample_point_on_envmap(rnd_param);
	}
	// TODO sphere sampling
	Shape shape = scene.shapes[light.shape_id];
	return sample_point_on_triangle_mesh(light.triangle_sampler, shape, ref_point, rnd_param);
}

/// Given a point on the light source and a reference point,
/// compute the sampling density for the function above.
float pdf_point_on_light(in Light light, PointAndNormal point_on_light, float3 ref_point)
{
    if (is_envmap(light)) {
        return pdf_point_on_envmap(point_on_light);
	}
    return 1.0 / light.surface_area;
}

/// Given a viewing direction pointing outwards from the light, and a point on the light,
/// compute the emission of the light. We also need the "footprint" of the ray
/// for texture filtering. For finite position, view_footprint stores (approximatedly) du/dx
/// and for infinite direction (e.g., envmap), view_footprint stores the approximated ddir/dx.
float3 emission(in Light light, float3 view_dir, float view_footprint, PointAndNormal point_on_light)
{
    if (is_envmap(light)) {
        return emission_envmap(view_dir, view_footprint);
	}
    return dot(point_on_light.normal, view_dir) <= 0 ? 0 : light.intensity;
}
