import shared;
import Scene;
import Camera;
import random;
import Intersection;
import Light;

import Utils;
import Frame;
import Texture;
import TableDist;
import Ray;

// ------------------------------------------------------------
// Ray Generation Shader (Path Tracing)
// ------------------------------------------------------------

#if 0
struct MaterialPayload {
    Material bsdf;
    PathVertex vertex;
    float3 dir_in;
    float pdf;
    float3 dir_out;
    float eta;
    float3 f_or_dir;
    float roughness;
};

[shader("callable")]
void eval_lambertian_bsdf(inout MaterialPayload payload)
{
}

[shader("callable")]
void sample_lambertian_bsdf(inout MaterialPayload payload)
{
}
#endif

#if 1
struct BSDFSampleRecord
{
    float3 dir_out;
    // The index of refraction ratio. Set to 0 if it's not a transmission event.
    float eta;
    float roughness; // Roughness of the selected BRDF layer ([0, 1]).
};
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

    return fmax(dot(frame.n, dir_out), 0) *
           eval(bsdf.reflectance, vertex.uv, vertex.uv_screen_size) / float.getPi();
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
Optional<BSDFSampleRecord> sample_bsdf(in Lambertian bsdf, in PathVertex vertex, float3 dir_in, float2 rnd_param)
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

    BSDFSampleRecord record = { to_world(frame, sample_cos_hemisphere(rnd_param)) };
    record.eta = 0;
    record.roughness = 1;
    return record;
}
#endif

[shader("raygeneration")]
void rgen_path_tracing()
{
    uint2 pixel_index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    pcg32_state rng = init_pcg32(frame.index * dimensions.x * dimensions.y + (pixel_index.y * dimensions.x + pixel_index.x));
    RayDesc ray = scene.camera.sample_primary(pixel_index, dimensions, rng);
    RayDifferential ray_diff = init_ray_differential(dimensions);

    float3 radiance = 0;
    Optional<PathVertex> vertex_ = intersect(ray);
    if (!vertex_.hasValue) {
		// Hit background. Account for the environment map if needed.
        if (scene.has_envmap) {
			radiance = emission_envmap(-ray.Direction, ray_diff.spread);
        }
        scene.accumulate_sample(pixel_index, float4(radiance, 1));
        return;
    }
    PathVertex vertex = vertex_.value;
		
	// A path's contribution is
    // C(v) = W(v0, v1) * G(v0, v1) * f(v0, v1, v2) *
    //                    G(v1, v2) * f(v1, v2, v3) *
    //                  ........
    //                  * G(v_{n-1}, v_n) * L(v_{n-1}, v_n)
    // where v is the path vertices, W is the sensor response
    // G is the geometry term, f is the BSDF, L is the emission
    //
    // "sample_primary" importance samples both W and G,
    // and we assume it always has weight 1.

    // current_path_throughput stores the ratio between
    // 1) the path contribution from v0 up to v_{i} (the BSDF f(v_{i-1}, v_i, v_{i+1}) is not included),
    // where i is where the PathVertex "vertex" lies on, and
    // 2) the probability density for computing the path v from v0 up to v_i,
    // so that we can compute the Monte Carlo estimates C/p.
    float3 current_path_throughput = 1;
    // eta_scale stores the scale introduced by Snell-Descartes law to the BSDF (eta^2).
    // We use the same Russian roulette strategy as Mitsuba/pbrt-v3
    // and tracking eta_scale and removing it from the
    // path contribution is crucial for many bounces of refraction.
    float eta_scale = 1;

    // We hit a light immediately.
    // This path has only two vertices and has contribution
    // C = W(v0, v1) * G(v0, v1) * L(v0, v1)
    if (vertex.area_light_id != -1) {
        radiance += current_path_throughput *
                    emission(vertex, -ray.Direction);
    }

    // We iteratively sum up path contributions from paths with different number of vertices
    // If max_depth == -1, we rely on Russian roulette for path termination.
    int max_depth = scene.options.max_depth;
    for (int num_vertices = 3; max_depth == -1 || num_vertices <= max_depth + 1; num_vertices++) {
        // We are at v_i, and all the path contribution on and before has been accounted for.
        // Now we need to somehow generate v_{i+1} to account for paths with more vertices.
        // In path tracing, we generate two vertices:
        // 1) we sample a point on the light source (often called "Next Event Estimation")
        // 2) we randomly trace a ray from the surface point at v_i and hope we hit something.
        //
        // The first importance samples L(v_i, v_{i+1}), and the second
        // importance samples f(v_{i-1}, v_i, v_{i+1}) * G(v_i, v_{i+1})
        //
        // We then combine the two sampling strategies to estimate the contribution using weighted average.
        // Say the contribution of the first sampling is C1 (with probability density p1),
        // and the contribution of the second sampling is C2 (with probability density p2,
        // then we compute the estimate as w1*C1/p1 + w2*C2/p2.
        //
        // Assuming the vertices for C1 is v^1, and v^2 for C2,
        // Eric Veach showed that it is a good idea setting
        // w1 = p_1(v^1)^k / (p_1(v^1)^k + p_2(v^1)^k)
        // w2 = p_2(v^2)^k / (p_1(v^2)^k + p_2(v^2)^k),
        // where k is some scalar real number, and p_a(v^b) is the probability density of generating
        // vertices v^b using sampling method "a".
        // We will set k=2 as suggested by Eric Veach.

        // Finally, we set our "next vertex" in the loop to the v_{i+1} generated
        // by the second sampling, and update current_path_throughput using
        // our hemisphere sampling.

        // Let's implement this!
        Lambertian mat = scene.materials[vertex.material_id];

        // First, we sample a point on the light source.
        // We do this by first picking a light source, then pick a point on it.
        float light_rnd = rng.next_float();
        int light_id = sample_light(light_rnd);
		float3 point_rnd = rng.next_float3();
        Light light = scene.lights[light_id];
        PointAndNormal point_on_light = sample_point_on_light(light, vertex.position, point_rnd);

        // Next, we compute w1*C1/p1. We store C1/p1 in C1.
        float3 C1 = 0;
        float w1 = 0;
        // Remember "current_path_throughput" already stores all the path contribution on and before v_i.
        // So we only need to compute G(v_{i}, v_{i+1}) * f(v_{i-1}, v_{i}, v_{i+1}) * L(v_{i}, v_{i+1})
        {
            // Let's first deal with C1 = G * f * L.
            // Let's first compute G.
            float G = 0;
            float3 dir_light;
            // The geometry term is different between directional light sources and
            // others. Currently we only have environment maps as directional light sources.
            if (!is_envmap(light)) {
                dir_light = normalize(point_on_light.position - vertex.position);
                // If the point on light is occluded, G is 0. So we need to test for occlusion.
                // To avoid self intersection, we need to set the tnear of the ray
                // to a small "epsilon". We set the epsilon to be a small constant times the
                // scale of the scene, which we can obtain through the get_shadow_epsilon() function.
                RayDesc shadow_ray;
                shadow_ray.Origin = vertex.position;
                shadow_ray.Direction = dir_light;
                shadow_ray.TMin = scene.shadow_epsilon();
				shadow_ray.TMax = (1 - scene.shadow_epsilon())
				                * distance(point_on_light.position, vertex.position);
                if (!occluded(shadow_ray)) {
                    // geometry term is cosine at v_{i+1} divided by distance squared
                    // this can be derived by the infinitesimal area of a surface projected on
                    // a unit sphere -- it's the Jacobian between the area measure and the solid angle
                    // measure.
                    G = max(-dot(dir_light, point_on_light.normal), 0) /
                        distance_squared(point_on_light.position, vertex.position);
                }
            } else {
                // The direction from envmap towards the point is stored in
                // point_on_light.normal.
                dir_light = -point_on_light.normal;
                // If the point on light is occluded, G is 0. So we need to test for occlusion.
                // To avoid self intersection, we need to set the tnear of the ray
                // to a small "epsilon" which we define as c_shadow_epsilon as a global constant.
                RayDesc shadow_ray;
                shadow_ray.Origin = vertex.position;
                shadow_ray.Direction = dir_light;
                shadow_ray.TMin = scene.shadow_epsilon();
                shadow_ray.TMax = float.maxValue; /* envmaps are infinitely far away */
                if (!occluded(shadow_ray)) {
                    // We integrate envmaps using the solid angle measure,
                    // so the geometry term is 1.
                    G = 1;
                }
            }

            // Before we proceed, we first compute the probability density p1(v1)
            // The probability density for light sampling to sample our point is
            // just the probability of sampling a light times the probability of sampling a point
            float p1 = light_pmf(light_id) *
                      pdf_point_on_light(light, point_on_light, vertex.position);

            // We don't need to continue the computation if G is 0.
            // Also sometimes there can be some numerical issue such that we generate
            // a light path with probability zero
            if (G > 0 && p1 > 0) {
                // Let's compute f (BSDF) next.
                float3 dir_view = -ray.Direction;
                // assert(vertex.material_id >= 0);
                float3 f = eval(mat, vertex, dir_view, dir_light);

                // Evaluate the emission
                // We set the footprint to zero since it is not fully clear how
                // to set it in this case.
                // One way is to use a roughness based heuristics, but we have multi-layered BRDFs.
                // See "Real-time Shading with Filtered Importance Sampling" from Colbert et al.
                // for the roughness based heuristics.
                float3 L = emission(light, -dir_light, 0, point_on_light);

                // C1 is just a product of all of them!
                C1 = G * f * L;

                // Next let's compute w1

                // Remember that we want to set
                // w1 = p_1(v^1)^2 / (p_1(v^1)^2 + p_2(v^1)^2)
                // Notice that all of the probability density share the same path prefix and those cancel out.
                // Therefore we only need to account for the generation of the vertex v_{i+1}.

                // The probability density for our hemispherical sampling to sample
                float p2 = pdf_sample_bsdf(mat, vertex, dir_view, dir_light);
                // !!!! IMPORTANT !!!!
                // In general, p1 and p2 now live in different spaces!!
                // our BSDF API outputs a probability density in the solid angle measure
                // while our light probability density is in the area measure.
                // We need to make sure that they are in the same space.
                // This can be done by accounting for the Jacobian of the transformation
                // between the two measures.
                // In general, I recommend to transform everything to area measure
                // (except for directional lights) since it fits to the path-space math better.
                // Converting a solid angle measure to an area measure is just a
                // multiplication of the geometry term G (let solid angle be dS, area be dA,
                // we have dA/dS = G).
                p2 *= G;

                w1 = (p1 * p1) / (p1 * p1 + p2 * p2);
                C1 /= p1;
            }
        }
        radiance += current_path_throughput * C1 * w1;

        // Let's do the hemispherical sampling next.
        float3 dir_view = -ray.Direction;
        float2 bsdf_rnd_param_uv  = rng.next_float2();
        float bsdf_rnd_param_w = rng.next_float();
        Optional<BSDFSampleRecord> bsdf_sample_ = sample_bsdf(mat,
						vertex,
                        dir_view,
                        bsdf_rnd_param_uv);
        if (!bsdf_sample_.hasValue) {
            // BSDF sampling failed. Abort the loop.
            break;
        }
        BSDFSampleRecord bsdf_sample = bsdf_sample_.value;
        float3 dir_bsdf = bsdf_sample.dir_out;
        // Update ray differentials & eta_scale
        if (bsdf_sample.eta == 0) {
            ray_diff.spread = reflect(ray_diff, vertex.mean_curvature, bsdf_sample.roughness);
        } else {
            ray_diff.spread = refract(ray_diff, vertex.mean_curvature, bsdf_sample.eta, bsdf_sample.roughness);
            eta_scale /= (bsdf_sample.eta * bsdf_sample.eta);
        }

        // Trace a ray towards bsdf_dir. Note that again we have
        // to have an "epsilon" tnear to prevent self intersection.
        RayDesc bsdf_ray;
        bsdf_ray.Origin = vertex.position;
        bsdf_ray.Direction = dir_bsdf;
        bsdf_ray.TMin = scene.intersection_epsilon();
        bsdf_ray.TMax = float.maxValue;
        Optional<PathVertex> bsdf_vertex = intersect(bsdf_ray);

        // To update current_path_throughput
        // we need to multiply G(v_{i}, v_{i+1}) * f(v_{i-1}, v_{i}, v_{i+1}) to it
        // and divide it with the pdf for getting v_{i+1} using hemisphere sampling.
        float G;
        if (bsdf_vertex.hasValue) {
            G = fabs(dot(dir_bsdf, bsdf_vertex.value.geometric_normal)) /
                distance_squared(bsdf_vertex.value.position, vertex.position);
        } else {
            // We hit nothing, set G to 1 to account for the environment map contribution.
            G = 1;
        }

        float3 f = eval(mat, vertex, dir_view, dir_bsdf);
        float p2 = pdf_sample_bsdf(mat, vertex, dir_view, dir_bsdf);
        if (p2 <= 0) {
            // Numerical issue -- we generated some invalid rays.
            break;
        }

        // Remember to convert p2 to area measure!
        p2 *= G;
        // note that G cancels out in the division f/p, but we still need
        // G later for the calculation of w2.

        // Now we want to check whether dir_bsdf hit a light source, and
        // account for the light contribution (C2 & w2 & p2).
        // There are two possibilities: either we hit an emissive surface,
        // or we hit an environment map.
        // We will handle them separately.
        if (bsdf_vertex.hasValue && bsdf_vertex.value.area_light_id != -1) {
            // G & f are already computed.
            float3 L = emission(bsdf_vertex.value, -dir_bsdf);
            float3 C2 = G * f * L;
            // Next let's compute p1(v2): the probability of the light source sampling
            // directly drawing the point corresponds to bsdf_dir.
            int light_id = bsdf_vertex.value.area_light_id;
            // assert(light_id >= 0);
            Light light = scene.lights[light_id];
            PointAndNormal light_point = {
				bsdf_vertex.value.position,
				bsdf_vertex.value.geometric_normal
			};
            float p1 = light_pmf(light_id) *
                      pdf_point_on_light(light, light_point, vertex.position);
            float w2 = (p2 * p2) / (p1 * p1 + p2 * p2);

            C2 /= p2;
            radiance += current_path_throughput * C2 * w2;
        } else if (!bsdf_vertex.hasValue && scene.has_envmap) {
            // G & f are already computed.
            float3 L = emission_envmap(
                                  -dir_bsdf, // pointing outwards from light
                                  ray_diff.spread);
            float3 C2 = G * f * L;
            // Next let's compute p1(v2): the probability of the light source sampling
            // directly drawing the direction bsdf_dir.
            PointAndNormal light_point = { float3(0), -dir_bsdf }; // pointing outwards from light
            float p1 = light_pmf(scene.envmap.light_id) *
                       pdf_point_on_envmap(light_point);
            float w2 = (p2 * p2) / (p1 * p1 + p2 * p2);

            C2 /= p2;
            radiance += current_path_throughput * C2 * w2;
        }

        if (!bsdf_vertex.hasValue) {
            // Hit nothing -- can't continue tracing.
            break;
        }

        // Update rays/intersection/current_path_throughput/current_pdf
        // Russian roulette heuristics
        float rr_prob = 1;
        if (num_vertices - 1 >= scene.options.rr_depth) {
            rr_prob = min(max_comp((1 / eta_scale) * current_path_throughput), 0.95);
            if (rng.next_float() > rr_prob) {
                // Terminate the path
                break;
            }
        }

        ray = bsdf_ray;
        vertex = bsdf_vertex.value;
        current_path_throughput = current_path_throughput * (G * f) / (p2 * rr_prob);
    }
	
    scene.accumulate_sample(pixel_index, float4(radiance, 1));
}
