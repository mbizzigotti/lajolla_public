import shared;
import Scene;
import Camera;
import random;
import Intersection;
import Light;

import Utils;
import Frame;

// ------------------------------------------------------------
// Ray Generation Shader (Path Tracing)
// ------------------------------------------------------------

float3 eval(in Lambertian bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Sometimes the shading normal can be inconsistent with
    // the geometry normal. We flip the shading frame in that
    // case so that we don't produces "black fringes".
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    return fmax(dot(frame.n, dir_out), 0) * bsdf.reflectance / float.getPi();
}

float pdf_sample_bsdf(in Lambertian bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // For Lambertian, we importance sample the cosine hemisphere domain.
    return fmax(dot(frame.n, dir_out), 0) / float.getPi();
}

Optional<float3> sample(in Lambertian bsdf, in PathVertex vertex, float3 dir_in, float2 rnd_param)
{
    // For Lambertian, we importance sample the cosine hemisphere domain.
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // Incoming direction is below the surface.
        return none;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    return to_world(frame, sample_cos_hemisphere(rnd_param));
}

[shader("raygeneration")]
void rgen_path_tracing()
{
    uint2 pixel_index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    pcg32_state rng = init_pcg32(frame.index * (pixel_index.y * dimensions.x + pixel_index.x));
    RayDesc ray = scene.camera.sample_primary(pixel_index, dimensions, rng);
    float3 radiance = 0;
    let vertex_ = intersect(ray);

    if (!vertex_.hasValue) {
        scene.accumulate_sample(pixel_index, float4(0));
	}

    PathVertex vertex = vertex_.value;
	
	if (vertex.area_light_id != -1) {
		radiance += emission(vertex, -ray.Direction);
	}

    float3 current_path_throughput = 1;
	
	int depth = 1;
	while (depth < 4)
    {
		float light_rnd = rng.next_float();
        int light_id = sample_light(light_rnd);
		float3 light_sample_rnd = rng.next_float3();
        Light light = scene.lights[light_id];
        PointAndNormal point_on_light = sample_point_on_light(light, vertex.position, light_sample_rnd);

        let material = scene.materials[vertex.material_id];
        float3 dir_view = -ray.Direction;
        float3 dir_light = normalize(point_on_light.position - vertex.position);
        RayDesc shadow_ray;
        shadow_ray.Origin = vertex.position + dir_light * scene.shadow_epsilon();
        shadow_ray.Direction = dir_light;
        shadow_ray.TMin = 0;
        shadow_ray.TMax = (1 - scene.shadow_epsilon()) * distance(point_on_light.position, vertex.position);
        if (!occluded(shadow_ray))
        {
            let G = max(-dot(dir_light, point_on_light.normal), 0)
			      / distance_squared(point_on_light.position, vertex.position);
            float p1 = light_pmf(light_id) *
                      pdf_point_on_light(light, point_on_light, vertex.position);
			float3 f = eval(material, vertex, dir_view, dir_light);
			float3 L = emission(light, -dir_light, 0, point_on_light);
            float3 C1 = G * f * L;
            C1 /= p1;
			radiance += C1;
        }

        Optional<float3> bsdf_sample = sample(material, vertex, dir_view, rng.next_float2());
        float3 dir_bsdf = bsdf_sample.value;

        // Trace a ray towards bsdf_dir. Note that again we have
        // to have an "epsilon" tnear to prevent self intersection.
        RayDesc bsdf_ray = { vertex.position, scene.intersection_epsilon(), dir_bsdf, 1e16 };
        Optional<PathVertex> bsdf_vertex = intersect(bsdf_ray);

        float G = 1;
        if (bsdf_vertex.hasValue) {
            G = fabs(dot(dir_bsdf, bsdf_vertex.value.geometric_normal)) /
                distance_squared(bsdf_vertex.value.position, vertex.position);
		}

        float3 f = eval(material, vertex, dir_view, dir_bsdf);
        float p2 = pdf_sample_bsdf(material, vertex, dir_view, dir_bsdf);

        if (bsdf_vertex.hasValue && bsdf_vertex.value.area_light_id != -1) {
            // G & f are already computed.
            float3 L = emission(bsdf_vertex.value, -dir_bsdf);
            float3 C2 = G * f * L;
            // Next let's compute p1(v2): the probability of the light source sampling
            // directly drawing the point corresponds to bsdf_dir.
            int light_id = bsdf_vertex.value.area_light_id;
            Light light = scene.lights[light_id];
            PointAndNormal light_point = { bsdf_vertex.value.position, bsdf_vertex.value.geometric_normal };
            float p1 = light_pmf(light_id) *
                      pdf_point_on_light(light, light_point, vertex.position);
            float w2 = (p2 * p2) / (p1 * p1 + p2 * p2);

            C2 /= p2;
            radiance += current_path_throughput * C2 * w2;
        }

        ray = bsdf_ray;
        vertex = bsdf_vertex.value;
        current_path_throughput = current_path_throughput * (G * f) / p2;
        depth += 1;
	}
    scene.accumulate_sample(pixel_index, float4(radiance, 1));
}
