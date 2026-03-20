import Shared;
import Scene;
import Camera;
import Random;
import Intersection;
import Light;
import Utils;
import Ray;
import Material;
import Medium;
import PhaseFunction;

//#define DEBUG_PIXEL (int2(436, 190)) // High on volume (NEE)
#define DEBUG_PIXEL (int2(320, 403)) // Low on volume (direct)
//#define DEBUG_PIXEL (int2(426, 168))
#if 0
#define NAN1_CHECK(x, ...) if (isnan(x))      printf("[NAN " #x "] " __VA_ARGS__)
#define NAN3_CHECK(x, ...) if (any(isnan(x))) printf("[NAN " #x "] " __VA_ARGS__)
#define DEBUG(...) if (all(DispatchRaysIndex().xy == DEBUG_PIXEL)) printf(__VA_ARGS__)
#else
#define NAN1_CHECK(x, ...) 0
#define NAN3_CHECK(x, ...) 0
#define DEBUG(...) 0
#endif

#define VOL_PATH_VERSION 5

// ------------------------------------------------------------
// Ray Generation Shader (Volume Path Tracing)
// ------------------------------------------------------------

[shader("raygeneration")]
void rgen_vol_path_tracing()
{
    uint2 pixel_index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    pcg32_state rng = init_pcg32(frame.index * dimensions.x * dimensions.y + (pixel_index.y * dimensions.x + pixel_index.x));
    RayDesc ray = scene.camera.sample_primary(pixel_index, dimensions, rng);
    RayDifferential ray_diff = init_ray_differential(dimensions);
    float3 radiance = 0;
	// TODO: Specialization Constant?
#if VOL_PATH_VERSION == -1
    radiance = debug_sigma_a(ray, ray_diff, rng);
#elif VOL_PATH_VERSION == 1
	radiance = vol_path_tracing_1(ray, ray_diff, rng);
#elif VOL_PATH_VERSION == 2
	radiance = vol_path_tracing_2(ray, ray_diff, rng);
#elif VOL_PATH_VERSION == 3
	radiance = vol_path_tracing_3(ray, ray_diff, rng);
#elif VOL_PATH_VERSION == 4
    radiance = vol_path_tracing_4(ray, ray_diff, rng);
#elif VOL_PATH_VERSION == 5
    radiance = vol_path_tracing_5(ray, ray_diff, rng);
#else
    radiance = vol_path_tracing(ray, ray_diff, rng);
#endif

    scene.accumulate_sample(pixel_index, float4(radiance, 1));
}

// This test makes a lot of assumptions about the scene!
float3 debug_sigma_a(RayDesc ray, in RayDifferential ray_diff, inout pcg32_state rng)
{
    Optional<PathVertex> surface_vertex = intersect(ray, ray_diff);
    if (!surface_vertex.hasValue) return 0;
	PathVertex vertex = surface_vertex.value;
    if (vertex.interior_medium_id == -1) return 0;
    ray.Origin = vertex.position;
    ray.TMin = scene.intersection_epsilon();
    ray.TMax = float.maxValue;
    Optional<PathVertex> medium_vertex = intersect(ray, ray_diff);
    if (!medium_vertex.hasValue) return 0;
    vertex = medium_vertex.value;

    Medium medium = scene.media[vertex.interior_medium_id];
    float t_max = distance(ray.Origin, vertex.position);
    float dt = 0.025;
    float3 radiance = 0; // "radiance"

	// Integrate sigma_a along the ray
    for (float t = 0; t < 2.0f; t += dt) {
        float3 p = ray.Origin + t * ray.Direction;
        float3 sigma_a = get_sigma_a(medium, p);
        radiance += sigma_a * dt;
    }

    return 20 * radiance;
}

[ForceInline]
int update_medium_id(in PathVertex vertex, in RayDesc ray,
					 int current_medium_id)
{
    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
        // At medium transition. Update medium id.
        if (dot(ray.Direction, vertex.geometric_normal) > 0) {
            current_medium_id = vertex.exterior_medium_id;
        } else {
            current_medium_id = vertex.interior_medium_id;
        }
    }
    return current_medium_id;
}

// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources
float3 vol_path_tracing_1(RayDesc ray, in RayDifferential ray_diff, inout pcg32_state rng)
{
    int current_medium_id = scene.camera.medium_id;

    Optional<PathVertex> vertex_ = intersect(ray, ray_diff);
    if (vertex_.hasValue) {
        PathVertex vertex = vertex_.value;
        if (vertex.area_light_id != -1) {
            float3 transmittance = 1;
            if (current_medium_id >= 0) {
                Medium medium = scene.media[current_medium_id];
                float3 p = ray.Origin;
                float3 sigma_a = get_sigma_a(medium, p);
                float3 sigma_s = get_sigma_s(medium, p);
                float3 sigma_t = sigma_s + sigma_a;
                float t = distance(ray.Origin, vertex.position);
                transmittance = exp(-sigma_t * t);
            }
            return transmittance * emission(vertex, -ray.Direction);
        }
    }
    return 0;
}

// The second simplest volumetric renderer:
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
float3 vol_path_tracing_2(RayDesc ray, in RayDifferential ray_diff, inout pcg32_state rng)
{
    // Homework 2: implememt this!
    int current_medium_id = scene.camera.medium_id;
    float3 radiance = 0;
    Optional<PathVertex> vertex_ = intersect(ray, ray_diff);
    if (current_medium_id >= 0) {
        Medium medium = scene.media[current_medium_id];
        // We have an integral \int_{0}^{t} T(t') sigma_s f G L dt' + T(max_t) Le
        // T(t) = exp(-\int_{0}^{t} sigma_t dt'')
        // We'll importance sample T
        float max_t = float.maxValue;
        if (vertex_.hasValue) {
            max_t = distance(vertex_.value.position, ray.Origin);
        }
        float3 sigma_a = get_sigma_a(medium, ray.Origin);
        float3 sigma_s = get_sigma_s(medium, ray.Origin);
        float3 sigma_t = sigma_s + sigma_a;

        // Sample T
        // We want to sample t s.t. p(t) ~ exp(-s * t)
        // We'll do the standard inverse transformation
        // first we integrate from 0 to t
        //   \int_{0}^{t} exp(-s * t') dt'
        // = -exp(-s * t) / s + 1/s
        // the normalization factor is thus 1/s (set t = infty)
        // p(t) = exp(-s * t) * s
        // the CDF is
        // P(t) = -exp(-s * t) + 1 = u
        // to invert the CDF:
        // exp(-s * t) = 1 - u
        // -m * t = log(1 - u)
        // t = log(1 - u) / -s

        // Assume monochromatic medium
        float t = -log(1 - rng.next_float()) / sigma_t[0];
        if (t < max_t) {
            // Direct lighting
            PathVertex vertex;
            vertex.position = ray.Origin + t * ray.Direction;
            vertex.interior_medium_id = current_medium_id;
            vertex.exterior_medium_id = current_medium_id;

            // float3 transmittance = exp(-sigma_t * t);
            // float pdf = exp(-sigma_t[0] * t) * sigma_t[0];
            // transmittance /= pdf;
            float3 transmittance = float(1) / sigma_t;
            PhaseFunction phase_function = medium.phase_function;

            // next event estimation
            float3 C1 = 0;
            {
                // Sample a point on the light source.
                float light_rnd = rng.next_float();
                int light_id = sample_light(light_rnd);
                float3 point_rnd = rng.next_float3();
                Light light = scene.lights[light_id];
                PointAndNormal point_on_light = sample_point_on_light(light, vertex.position, point_rnd);
                float G = 0;
                float3 dir_light = normalize(point_on_light.position - vertex.position);
                RayDesc shadow_ray;
                shadow_ray.Origin = vertex.position;
                shadow_ray.Direction = dir_light;
                shadow_ray.TMin = scene.shadow_epsilon();
                shadow_ray.TMax = (1 - scene.shadow_epsilon())
				                * distance(point_on_light.position, vertex.position);
                if (!occluded(shadow_ray)) {
                    G = max(-dot(dir_light, point_on_light.normal), float(0)) /
                        distance_squared(point_on_light.position, vertex.position);
                }
                float p1 = light_pmf(light_id) *
                          pdf_point_on_light(light, point_on_light, vertex.position);
                if (G > 0 && p1 > 0) {
                    // Compute sigma_s * T * G * f * L
                    float3 dir_view = -ray.Direction;
                    float3 f = eval(phase_function, dir_view, dir_light);
                    float3 L = emission(light, -dir_light, float(0), point_on_light);
                    float3 T = exp(-sigma_t * distance(vertex.position, point_on_light.position));
                    C1 = sigma_s * transmittance * T * G * f * L / p1;
                }
            }
            radiance += C1;
        } else {
            // float3 transmittance = exp(-sigma_t * max_t);
            // float pdf = exp(-sigma_t[0] * max_t);
            // transmittance /= pdf;
            if (vertex_.hasValue) {
                if (vertex_.value.area_light_id != -1) {
                    radiance += emission(vertex_.value, -ray.Direction);
                }
            }
        }
    }
    return radiance;
}

// The third volumetric renderer (not so simple anymore):
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
float3 vol_path_tracing_3(RayDesc ray, in RayDifferential ray_diff, inout pcg32_state rng)
{
    int current_medium_id = scene.camera.medium_id;
    float3 radiance = 0;
    int max_depth = scene.options.max_depth;
    float3 current_path_throughput = 1;
    for (int bounces = 0; max_depth == -1 || bounces < max_depth; bounces++) {
        Optional<PathVertex> surface_vertex = intersect(ray, ray_diff);
        bool scatter = false;
        PathVertex vertex;
        if (surface_vertex.hasValue) {
            vertex = surface_vertex.value;
        }

        // Consider medium transmittance by sampling a distance along the medium.
        // Update "vertex". Compute a transmittance and its PDF.
        float3 transmittance = 1;
        float trans_pdf = 1;
        if (current_medium_id >= 0) {
            Medium medium = scene.media[current_medium_id];
            // We have an integral \int_{0}^{t} T(t') sigma_s f G L dt' + T(max_t) Le
            // T(t) = exp(-\int_{0}^{t} sigma_t dt'')
            // We'll importance sample T
            float max_t = float.maxValue;
            if (surface_vertex.hasValue) {
                max_t = distance(vertex.position, ray.Origin);
            }
            float3 sigma_a = get_sigma_a(medium, ray.Origin);
            float3 sigma_s = get_sigma_s(medium, ray.Origin);
            float3 sigma_t = sigma_s + sigma_a;

            // Assume monochromatic medium
            float t = -log(1 - rng.next_float()) / sigma_t[0];
            if (t < max_t) {
                scatter = true; // Tell the code to sample the phase function later

                // Update vertex information
                vertex.position = ray.Origin + t * ray.Direction;
                vertex.interior_medium_id = current_medium_id;
                vertex.exterior_medium_id = current_medium_id;

                transmittance = exp(-sigma_t * t);
                trans_pdf = exp(-sigma_t[0] * t) * sigma_t[0];
            } else { // t >= max_t
                if (!surface_vertex.hasValue) {
                    // do we need this?
                    vertex.position = ray.Origin + max_t * ray.Direction;
                    vertex.interior_medium_id = current_medium_id;
                    vertex.exterior_medium_id = current_medium_id;
                }
                transmittance = exp(-sigma_t * max_t);
                trans_pdf = exp(-sigma_t[0] * max_t);
            }
        }

        // Update current path throughput with transmittance
        current_path_throughput *= (transmittance / trans_pdf);

        // If we reach a surface and didn't scatter, include the emission.
        if (!scatter) {
            if (surface_vertex.hasValue) {
                if (vertex.area_light_id != -1) {
                    // current_path_throughput already accounts for transmittance/pdf.
                    radiance += current_path_throughput *
                                emission(vertex, -ray.Direction);
                }
            }
        }

        if (max_depth != -1 && bounces == max_depth - 1) {
            break;
        }

        // If we reach a surface and it is index-matching, skip through it
        // if we hit nothing and we're not scattering, terminate.
        if (!scatter) {
            if (surface_vertex.hasValue) {
                if (vertex.material_id == -1) {
                    ray = scene.make_ray(vertex.position, ray.Direction);
                    current_medium_id =
                        update_medium_id(vertex, ray, current_medium_id);
                    continue;
                }
            }
        }

        // Sample the next direction & update current_path_throughput
        float3 next_dir;
        if (scatter) {
            // Phase function sampling
            // assert(current_medium_id >= 0);
            Medium medium = scene.media[current_medium_id];
            PhaseFunction phase_function = medium.phase_function;
            float3 dir_view = -ray.Direction;
            float2 phase_rnd_param = rng.next_float2();
            Optional<float3> phase_sample =
                sample_phase_function(phase_function,
                                      dir_view,
                                      phase_rnd_param);
            if (!phase_sample.hasValue) {
                // Phase function sampling failed. Abort the loop.
                break;
            }
            next_dir = phase_sample.value;
            float3 f = eval(phase_function, dir_view, next_dir);
            float p2 = pdf_sample_phase(phase_function, dir_view, next_dir);
            // Need to remember multiplying the scattering coefficient!
            float3 sigma_s = get_sigma_s(medium, vertex.position);
            current_path_throughput *= sigma_s * (f / p2);
        } else {
            // surface case: don't need to deal with this yet.
            break;
        }

        // Update rays/current_path_throughput/current_pdf
        // Russian roulette heuristics
        float rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max_comp(current_path_throughput), 0.95);
            if (rng.next_float() > rr_prob) {
                // Terminate the path
                break;
            }
            current_path_throughput /= rr_prob;
        }

        // Update rays
        ray = scene.make_ray(vertex.position, next_dir);
        current_medium_id =
            update_medium_id(vertex, ray, current_medium_id);
    }
    return radiance;
}

// The fourth volumetric renderer:
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
float3 vol_path_tracing_4(RayDesc ray, in RayDifferential ray_diff, inout pcg32_state rng)
{
    int current_medium_id = scene.camera.medium_id;
    float3 radiance = 0;
    int max_depth = scene.options.max_depth;
    float3 current_path_throughput = 1;
    // To add contribution from phase function sampling (or directional sampling)
    // and combine it with next event estimation using MIS,
    // we need to keep track of the PDFs for both of them.
    // For directional sampling, we need to account for the phase function/BSDF sampling PDF.
    // For next event estimation, we need to account for the path vertex that
    // issued the light source sampling and went through multiple index-matching interfaces.
    // For both sampling strategies, we need to account for the probability of
    // the transmittance sampling going through multiple index-matching interfaces.
    // All quantities needs to be accessed across multiple bounces,
    // For the direction sampling PDF, we store it in dir_pdf.
    // For the next event estimation path vertex, we store it in nee_p_cache.
    // For the transmittance sampling probability, we store it in multi_trans_pdf.
    float dir_pdf = 0; // in solid angle measure
    float3 nee_p_cache;
    float multi_trans_pdf = 1;
    bool never_scatter = true;
    for (int bounces = 0; max_depth == -1 || bounces < max_depth; bounces++) {
        Optional<PathVertex> surface_vertex = intersect(ray, ray_diff);
        bool scatter = false;
        PathVertex vertex;
        if (surface_vertex.hasValue) {
            vertex = surface_vertex.value;
        }

        // Consider medium transmittance by sampling a distance along the medium.
        // Update "vertex". Compute a transmittance and its PDF.
        float3 transmittance = 1;
        float trans_pdf = 1;
        if (current_medium_id >= 0) {
            Medium medium = scene.media[current_medium_id];
            // We have an integral \int_{0}^{t} T(t') sigma_s f G L dt' + T(max_t) Le
            // T(t) = exp(-\int_{0}^{t} sigma_t dt'')
            // We'll importance sample T
            float max_t = float.maxValue;
            if (surface_vertex.hasValue) {
                max_t = distance(surface_vertex.value.position, ray.Origin);
            }
            float3 sigma_a = get_sigma_a(medium, ray.Origin);
            float3 sigma_s = get_sigma_s(medium, ray.Origin);
            float3 sigma_t = sigma_s + sigma_a;

            // Assume monochromatic medium
            float t = -log(1 - rng.next_float()) / sigma_t[0];
            if (t < max_t) {
                scatter = true; // Tell the code to sample the phase function later
                never_scatter = false;

                // Update vertex information
                vertex.position = ray.Origin + t * ray.Direction;
                vertex.interior_medium_id = current_medium_id;
                vertex.exterior_medium_id = current_medium_id;

                transmittance = exp(-sigma_t * t);
                trans_pdf = exp(-sigma_t[0] * t) * sigma_t[0];
            } else { // t >= max_t
                if (!surface_vertex.hasValue) {
                    // do we need this?
                    vertex.position = ray.Origin + max_t * ray.Direction;
                    vertex.interior_medium_id = current_medium_id;
                    vertex.exterior_medium_id = current_medium_id;
                }
                transmittance = exp(-sigma_t * max_t);
                trans_pdf = exp(-sigma_t[0] * max_t);
            }
        }
        // Update multiple bounces transmittion pdf.
        multi_trans_pdf *= trans_pdf;

        // Update current path throughput with transmittance
        current_path_throughput *= (transmittance / trans_pdf);

        // If we didn't scatter and didn't reach a surface either,
        // we're done
        if (!scatter && !surface_vertex.hasValue) {
            break;
        }

        // If we reach a surface and didn't scatter, include the emission.
        if (!scatter) {
            if (surface_vertex.hasValue) {
                if (vertex.area_light_id != -1) {
                    if (never_scatter) {
                        // This is the only way we can see the light source, so
                        // we don't need multiple importance sampling.
                        radiance += current_path_throughput *
                                    emission(vertex, -ray.Direction);
                    } else {
                        // Need to account for next event estimation
                        int light_id = vertex.area_light_id;
                        // assert(light_id >= 0);
                        Light light = scene.lights[light_id];
                        PointAndNormal light_point = { vertex.position, vertex.geometric_normal };
                        // Note that p_nee needs to account for the path vertex that issued
                        // next event estimation potentially many bounces ago.
                        // The vertex position is stored in nee_p_cache.
                        float p_nee = light_pmf(light_id) *
                                     pdf_point_on_light(light, light_point, nee_p_cache);
                        // The PDF for sampling the light source using directional sampling + transmittance sampling
                        // The directional sampling pdf was cached in dir_pdf in solid angle measure.
                        // The transmittance sampling pdf was cached in multi_trans_pdf.
                        float3 light_dir = normalize(vertex.position - nee_p_cache);
                        float G = fabs(dot(vertex.geometric_normal, light_dir)) /
                                 distance_squared(nee_p_cache, vertex.position);
                        float p_dir = dir_pdf * multi_trans_pdf * G;
                        float w2 = (p_dir * p_dir) / (p_dir * p_dir + p_nee * p_nee);
                        // current_path_throughput already accounts for prev_dir_pdf & transmittance.
                        radiance += current_path_throughput *
                                    emission(vertex, -ray.Direction) * w2;
                    }
                    return radiance;
                }
            }
        }

        if (max_depth != -1 && bounces == max_depth - 1) {
            break;
        }

        // If we reach a surface and it is index-matching, skip through it
        // if we hit nothing and we're not scattering, terminate.
        if (!scatter) {
            if (surface_vertex.hasValue) {
                if (surface_vertex.value.material_id == -1) {
                    ray.Origin = vertex.position;
                    ray.TMin = scene.intersection_epsilon();
                    ray.TMax = float.maxValue;
                    current_medium_id =
                        update_medium_id(vertex, ray, current_medium_id);
                    continue;
                }
            }
        }

        // Cache the NEE vertex for later use
        nee_p_cache = vertex.position;
        // We have a scattering event, reset multi_trans_pdf
        multi_trans_pdf = 1;

        // next event estimation
        float3 C1 = 0;
        float w1 = 0;
        {
            // Sample a point on the light source.
			float light_rnd = rng.next_float();
			int light_id = sample_light(light_rnd);
			float3 point_rnd = rng.next_float3();
			Light light = scene.lights[light_id];
			PointAndNormal point_on_light =
				sample_point_on_light(light, vertex.position, point_rnd);
            // Compute transmittance to light. Skip through index-matching shapes.
            float3 T_light = 1;
            float3 p = vertex.position;
            int shadow_medium_id = current_medium_id;
            int shadow_bounces = 0;
            float p_trans_dir = 1;
            while (true) {
                float3 dir_light = normalize(point_on_light.position - p);
                RayDesc shadow_ray;
                shadow_ray.Origin = p;
                shadow_ray.Direction = dir_light;
                shadow_ray.TMin = scene.shadow_epsilon();
                shadow_ray.TMax = (1 - scene.shadow_epsilon())
				                * distance(point_on_light.position, p);
                Optional<PathVertex> shadow_vertex = intersect(shadow_ray);
                float next_t = shadow_ray.TMax;
                if (shadow_vertex.hasValue) {
                    next_t = distance(p, shadow_vertex.value.position);
                }

                // Account for the transmittance to next_t
                if (shadow_medium_id >= 0) {
                    Medium medium = scene.media[shadow_medium_id];
                    float3 sigma_a = get_sigma_a(medium, ray.Origin);
                    float3 sigma_s = get_sigma_s(medium, ray.Origin);
                    float3 sigma_t = sigma_s + sigma_a;
                    T_light *= exp(-sigma_t * next_t);
                    p_trans_dir *= exp(-sigma_t[0] * next_t);
                }

                if (!shadow_vertex.hasValue) {
                    // Nothing is blocking, we're done
                    break;
                } else {
                    // Something is blocking: is it an opaque surface?
                    if (shadow_vertex.value.material_id >= 0) {
                        // we're blocked
                        T_light = 0;
                        break;
                    }
                    // otherwise, we want to pass through -- this introduces
                    // one extra connection vertex
                    shadow_bounces++;
                    if (max_depth != -1 && bounces + shadow_bounces + 1 >= max_depth) {
                        // Reach the max no. of vertices
                        T_light = 0;
                        break;
                    }
                    // let's update and continue
                    shadow_medium_id = update_medium_id(shadow_vertex.value, shadow_ray, shadow_medium_id);
                    p = p + next_t * dir_light;
                }
            }

            if (max_comp(T_light) > 0) {
                // Compute sigma_s * T * T_light * G * f * L
                float3 dir_light = normalize(point_on_light.position - vertex.position);
                float G = max(-dot(dir_light, point_on_light.normal), float(0)) /
                          distance_squared(point_on_light.position, vertex.position);
                if (isnan(G)) {
                    printf("G: [%v3f][%v3f] [%v3f][%v3f]", dir_light, point_on_light.normal, point_on_light.position, vertex.position);
                }
                float p1 = light_pmf(light_id) *
                          pdf_point_on_light(light, point_on_light, vertex.position);
                float3 dir_view = -ray.Direction;
                float3 f;
                // are we on a surface or are we in a medium?
                if (scatter) {
                    // assert(current_medium_id >= 0);
                    Medium medium = scene.media[current_medium_id];
                    PhaseFunction phase_function = medium.phase_function;
                    f = eval(phase_function, dir_view, dir_light);
                } else {
                    Material mat = scene.materials[vertex.material_id];
                    f = eval(mat, vertex, dir_view, dir_light);
                }
                float3 sigma_s = 1;
                if (scatter) {
                    Medium medium = scene.media[current_medium_id];
                    sigma_s = get_sigma_s(medium, vertex.position);
                }
                float3 L = emission(light, -dir_light, float(0), point_on_light);
                C1 = current_path_throughput * sigma_s * T_light * G * f * L / p1;
                // Multiple importance sampling: it's also possible
                // that a phase function sampling + multiple exponential sampling
                // will reach the light source.
                // We also need to multiply with G to convert phase function PDF to area measure.
                float p2 = 0;
                if (scatter) {
                    // assert(current_medium_id >= 0);
                    Medium medium = scene.media[current_medium_id];
                    PhaseFunction phase_function = medium.phase_function;
                    p2 = pdf_sample_phase(phase_function, dir_view, dir_light) * G;
                } else {
                    // BRDF importance sampling, no need to deal with this yet
                }
                p2 *= p_trans_dir;
                w1 = (p1 * p1) / (p1 * p1 + p2 * p2);
            }
        }
        radiance += C1 * w1;
        // if (any(isnan(C1))) {
        //     printf("NEE: [%v3f][%f]", C1, w1);
        // }

        // Sample the next direction & update current_path_throughput
        float3 next_dir;
        if (scatter) {
            // Phase function sampling
            // assert(current_medium_id >= 0);
            Medium medium = scene.media[current_medium_id];
            PhaseFunction phase_function = medium.phase_function;
            float3 dir_view = -ray.Direction;
            float2 phase_rnd_param = rng.next_float2();
            Optional<float3> phase_sample =
                sample_phase_function(phase_function, dir_view,
                                      phase_rnd_param);
            if (!phase_sample.hasValue) {
                // Phase function sampling failed. Abort the loop.
                break;
            }
            next_dir = phase_sample.value;
            float3 f = eval(phase_function, dir_view, next_dir);
            float p2 = pdf_sample_phase(phase_function, dir_view, next_dir);
            dir_pdf = p2;
            // Need to remember multiplying the scattering coefficient!
            float3 sigma_s = get_sigma_s(medium, vertex.position);
            current_path_throughput *= sigma_s * (f / p2);
        } else {
            // surface case: don't need to deal with this yet.
            break;
        }

        // Update rays/current_path_throughput/current_pdf
        // Russian roulette heuristics
        float rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max_comp(current_path_throughput), 0.95);
            if (rng.next_float() > rr_prob) {
                // Terminate the path
                break;
            }
            current_path_throughput /= rr_prob;
        }

        // Update rays
        ray = scene.make_ray(vertex.position, next_dir);
        current_medium_id =
            update_medium_id(vertex, ray, current_medium_id);
    }
    return radiance;
}

// The fifth volumetric renderer:
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
float3 vol_path_tracing_5(RayDesc ray, in RayDifferential ray_diff, inout pcg32_state rng)
{
    int current_medium_id = scene.camera.medium_id;
    float3 radiance = 0;
    int max_depth = scene.options.max_depth;
    float3 current_path_throughput = 1;
    // To add contribution from phase function sampling (or directional sampling)
    // and combine it with next event estimation using MIS,
    // we need to keep track of the PDFs for both of them.
    // For directional sampling, we need to account for the phase function/BSDF sampling PDF.
    // For next event estimation, we need to account for the path vertex that
    // issued the light source sampling and went through multiple index-matching interfaces.
    // For both sampling strategies, we need to account for the probability of
    // the transmittance sampling going through multiple index-matching interfaces.
    // All quantities needs to be accessed across multiple bounces,
    // For the direction sampling PDF, we store it in dir_pdf.
    // For the next event estimation path vertex, we store it in nee_p_cache.
    // For the transmittance sampling probability, we store it in multi_trans_pdf.
    float dir_pdf = 0; // in solid angle measure
    float3 nee_p_cache;
    float multi_trans_pdf = 1;
    bool never_scatter = true;
    for (int bounces = 0; max_depth == -1 || bounces < max_depth; bounces++) {
        Optional<PathVertex> surface_vertex = intersect(ray, ray_diff);
        bool scatter = false;
        PathVertex vertex;
        if (surface_vertex.hasValue) {
            vertex = surface_vertex.value;
        }

        // Consider medium transmittance by sampling a distance along the medium.
        // Update "vertex". Compute a transmittance and its PDF.
        float3 transmittance = 1;
        float trans_pdf = 1;
        if (current_medium_id >= 0) {
            Medium medium = scene.media[current_medium_id];
            // We have an integral \int_{0}^{t} T(t') sigma_s f G L dt' + T(max_t) Le
            // T(t) = exp(-\int_{0}^{t} sigma_t dt'')
            // We'll importance sample T
            float max_t = float.maxValue;
            if (surface_vertex.hasValue) {
                max_t = distance(surface_vertex.value.position, ray.Origin);
            }
            float3 sigma_a = get_sigma_a(medium, ray.Origin);
            float3 sigma_s = get_sigma_s(medium, ray.Origin);
            float3 sigma_t = sigma_s + sigma_a;

            // Assume monochromatic medium
            float t = -log(1 - rng.next_float()) / sigma_t[0];
            if (t < max_t) {
                scatter = true; // Tell the code to sample the phase function later
                never_scatter = false;

                // Update vertex information
                vertex.position = ray.Origin + t * ray.Direction;
                vertex.interior_medium_id = current_medium_id;
                vertex.exterior_medium_id = current_medium_id;

                transmittance = exp(-sigma_t * t);
                trans_pdf = exp(-sigma_t[0] * t) * sigma_t[0];
            } else { // t >= max_t
                if (!surface_vertex.hasValue) {
                    // do we need this?
                    vertex.position = ray.Origin + max_t * ray.Direction;
                    vertex.interior_medium_id = current_medium_id;
                    vertex.exterior_medium_id = current_medium_id;
                }
                transmittance = exp(-sigma_t * max_t);
                trans_pdf = exp(-sigma_t[0] * max_t);
            }
        }
        // Update multiple bounces transmittion pdf.
        multi_trans_pdf *= trans_pdf;

        // Update current path throughput with transmittance
        current_path_throughput *= (transmittance / trans_pdf);
        DEBUG("UPDATE (%v3f) %v3f %f [SCATTER %i] [VERTEX %i]", current_path_throughput, transmittance, trans_pdf, scatter, surface_vertex.hasValue);

        // If we didn't scatter and didn't reach a surface either,
        // we're done
        if (!scatter && !surface_vertex.hasValue) {
            break;
        }

        // If we reach a surface and didn't scatter, include the emission.
        if (!scatter) {
            if (surface_vertex.hasValue) {
                if (vertex.area_light_id != -1) {
                    if (never_scatter) {
                        // This is the only way we can see the light source, so
                        // we don't need multiple importance sampling.
                        radiance += current_path_throughput *
                                    emission(vertex, -ray.Direction);
                    } else {
                        // Need to account for next event estimation
                        int light_id = vertex.area_light_id;
                        // assert(light_id >= 0);
                        Light light = scene.lights[light_id];
                        PointAndNormal light_point = { vertex.position, vertex.geometric_normal };
                        // Note that p_nee needs to account for the path vertex that issued
                        // next event estimation potentially many bounces ago.
                        // The vertex position is stored in nee_p_cache.
                        float p_nee = light_pmf(light_id) *
                                     pdf_point_on_light(light, light_point, nee_p_cache);
                        // The PDF for sampling the light source using directional sampling + transmittance sampling
                        // The directional sampling pdf was cached in dir_pdf in solid angle measure.
                        // The transmittance sampling pdf was cached in multi_trans_pdf.
                        float3 light_dir = normalize(vertex.position - nee_p_cache);
                        float G = fabs(dot(vertex.geometric_normal, light_dir)) /
                                 distance_squared(nee_p_cache, vertex.position);
                        float p_dir = dir_pdf * multi_trans_pdf * G;
                        float w2 = (p_dir * p_dir) / (p_dir * p_dir + p_nee * p_nee);
                        // current_path_throughput already accounts for prev_dir_pdf & transmittance.
                        radiance += current_path_throughput *
                                    emission(vertex, -ray.Direction) * w2;
                    }
                    return radiance;
                }
            }
        }

        if (max_depth != -1 && bounces == max_depth - 1) {
            break;
        }

        // If we reach a surface and it is index-matching, skip through it
        // if we hit nothing and we're not scattering, terminate.
        if (!scatter) {
            if (surface_vertex.value.material_id == -1) {
                ray = scene.make_ray(vertex.position, ray.Direction);
                current_medium_id = update_medium_id(vertex, ray, current_medium_id);
                continue;
            }
        }

        // Cache the NEE vertex for later use
        nee_p_cache = vertex.position;
        // We have a scattering event (medium or surface), reset multi_trans_pdf
        multi_trans_pdf = 1;

        // next event estimation
        float3 C1 = 0;
        float w1 = 0;
        {
            // Sample a point on the light source.
            float light_rnd = rng.next_float();
            int light_id = sample_light(light_rnd);
            float3 point_rnd = rng.next_float3();
            Light light = scene.lights[light_id];
            PointAndNormal point_on_light = sample_point_on_light(light, vertex.position, point_rnd);
            DEBUG("NEE (%v3f)(%v3f) %v3f", point_on_light.position, point_on_light.normal, vertex.position);
            // Compute transmittance to light. Skip through index-matching shapes.
            float3 T_light = 1;
            float3 p = vertex.position;
            int shadow_medium_id = current_medium_id;
            int shadow_bounces = 0;
            float p_trans_dir = 1;
            while (true) {
                float3 dir_light = normalize(point_on_light.position - p);
                RayDesc shadow_ray;
                shadow_ray.Origin = p;
                shadow_ray.Direction = dir_light;
                shadow_ray.TMin = scene.shadow_epsilon();
                shadow_ray.TMax = (1 - scene.shadow_epsilon())
				                * distance(point_on_light.position, p);
                Optional<PathVertex> shadow_vertex = intersect(shadow_ray);
                float next_t = shadow_ray.TMax;
                if (shadow_vertex.hasValue) {
                    next_t = distance(p, shadow_vertex.value.position);
                }

                // Account for the transmittance to next_t
                if (shadow_medium_id >= 0) {
                    Medium medium = scene.media[shadow_medium_id];
                    float3 sigma_a = get_sigma_a(medium, ray.Origin);
                    float3 sigma_s = get_sigma_s(medium, ray.Origin);
                    float3 sigma_t = sigma_s + sigma_a;
                    T_light *= exp(-sigma_t * next_t);
                    p_trans_dir *= exp(-sigma_t[0] * next_t);
                }

                if (!shadow_vertex.hasValue) {
                    // Nothing is blocking, we're done
                    break;
                } else {
                    // Something is blocking: is it an opaque surface?
                    if (shadow_vertex.value.material_id >= 0) {
                        // we're blocked
                        T_light = 0;
                        p_trans_dir = 0;
                        break;
                    }
                    // otherwise, we want to pass through -- this introduces
                    // one extra connection vertex
                    shadow_bounces++;
                    if (max_depth != -1 && bounces + shadow_bounces + 1 >= max_depth) {
                        // Reach the max no. of vertices
                        T_light = 0;
                        break;
                    }
                    // let's update and continue
                    shadow_medium_id = update_medium_id(shadow_vertex.value, shadow_ray, shadow_medium_id);
                    p = p + next_t * dir_light;
                }
            }

            if (max_comp(T_light) > 0) {
                // Compute sigma_s * T * T_light * G * f * L
                float3 dir_light = normalize(point_on_light.position - vertex.position);
                float G = max(-dot(dir_light, point_on_light.normal), float(0)) /
                         distance_squared(point_on_light.position, vertex.position);
                float p1 = light_pmf(light_id) *
                          pdf_point_on_light(light, point_on_light, vertex.position);
                float3 dir_view = -ray.Direction;
                float3 f;
                // are we on a surface or are we in a medium?
                if (scatter) {
                    // assert(current_medium_id >= 0);
                    Medium medium = scene.media[current_medium_id];
                    PhaseFunction phase_function = medium.phase_function;
                    f = eval(phase_function, dir_view, dir_light);
                } else {
                    Material mat = scene.materials[vertex.material_id];
                    f = eval(mat, vertex, dir_view, dir_light);
                }
                float3 sigma_s = 1;
                if (scatter) {
                    Medium medium = scene.media[current_medium_id];
                    sigma_s = get_sigma_s(medium, vertex.position);
                }
                float3 L = emission(light, -dir_light, float(0), point_on_light);
                C1 = current_path_throughput * sigma_s * T_light * G * f * L / p1;
                // Multiple importance sampling: it's also possible
                // that a phase function sampling + multiple steps exponential sampling
                // will reach the light source.
                // The probability for multiple steps exponential sampling
                // is stored in p_trans_dir
                // We also need to multiply with G to convert phase function PDF to area measure.
                float p2 = 0;
                if (scatter) {
                    // assert(current_medium_id >= 0);
                    Medium medium = scene.media[current_medium_id];
                    PhaseFunction phase_function = medium.phase_function;
                    p2 = pdf_sample_phase(phase_function, dir_view, dir_light) * G;
                } else {
                    // assert(vertex.material_id >= 0);
                    Material mat = scene.materials[vertex.material_id];
                    p2 = pdf_sample_bsdf(mat, vertex, dir_view, dir_light) * G;
                }
                p2 *= p_trans_dir;
                w1 = (p1 * p1) / (p1 * p1 + p2 * p2);
                DEBUG("NEE CONTRIB (%v3f)(%f) P1 %f P2 %f", C1, w1, p1, p2);
            }
        }
        radiance += C1 * w1;

        // Sample the next direction & update current_path_throughput
        float3 next_dir;
        if (scatter) {
            // Phase function sampling
            // assert(current_medium_id >= 0);
            Medium medium = scene.media[current_medium_id];
            PhaseFunction phase_function = medium.phase_function;
            float3 dir_view = -ray.Direction;
            float2 phase_rnd_param = rng.next_float2();
            Optional<float3> phase_sample =
                sample_phase_function(phase_function, dir_view,
                                      phase_rnd_param);
            if (!phase_sample.hasValue) {
                // Phase function sampling failed. Abort the loop.
                break;
            }
            next_dir = phase_sample.value;
            float3 f = eval(phase_function, dir_view, next_dir);
            float p2 = pdf_sample_phase(phase_function, dir_view, next_dir);
            dir_pdf = p2;
            // Need to remember multiplying the scattering coefficient!
            float3 sigma_s = get_sigma_s(medium, vertex.position);
            current_path_throughput *= sigma_s * (f / p2);
        } else {
            // BSDF sampling
            Material mat = scene.materials[vertex.material_id];
            float3 dir_view = -ray.Direction;
            float3 bsdf_rnd_param = rng.next_float3();
            Optional<BSDFSampleRecord> bsdf_sample =
                sample_bsdf(mat, vertex, dir_view, bsdf_rnd_param);
            if (!bsdf_sample.hasValue) {
                // Phase function sampling failed. Abort the loop.
                break;
            }
            next_dir = bsdf_sample.value.dir_out;
            float3 f = eval(mat, vertex, dir_view, next_dir);
            float p2 = pdf_sample_bsdf(mat, vertex, dir_view, next_dir);
            dir_pdf = p2;
            // Need to remember multiplying the scattering coefficient!
            current_path_throughput *= (f / p2);
            NAN3_CHECK(current_path_throughput, "[%v3f][%v3f][%v3f][%v3f]", dir_view, next_dir, vertex.shading_frame.n, vertex.geometric_normal);
        }
        never_scatter = false;

        // Update rays/current_path_throughput/current_pdf
        // Russian roulette heuristics
        float rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max_comp(current_path_throughput), 0.95);
            if (rng.next_float() > rr_prob) {
                // Terminate the path
                break;
            }
            current_path_throughput /= rr_prob;
        }

        // Update rays
        ray = scene.make_ray(vertex.position, next_dir);
        current_medium_id =
            update_medium_id(vertex, ray, current_medium_id);
    }
    return radiance;
}

float3 vol_path_tracing(RayDesc ray, in RayDifferential ray_diff, inout pcg32_state rng)
{
    int current_medium_id = scene.camera.medium_id;
    float3 radiance = 0;
    int max_depth = scene.options.max_depth;
    float3 current_path_throughput = 1;
    float dir_pdf = 0; // in solid angle measure
    float3 nee_p_cache;
    float3 multi_trans_dir_pdf = 1;
    float3 multi_trans_nee_pdf = 1;
    bool never_scatter = true;
    for (int bounces = 0; max_depth == -1 || bounces < max_depth; bounces++) {
        Optional<PathVertex> surface_vertex = intersect(ray, ray_diff);
        bool scatter = false;
        PathVertex vertex;
        if (surface_vertex.hasValue) {
            vertex = surface_vertex.value;
        }

        // Consider medium transmittance by sampling a distance along the medium.
        // Update "vertex". Compute a transmittance and its PDF.
        float3 transmittance = 1;
        // Now we have different transmittance sampling strategy for each channel,
        // so the PDF is a float3 instead of a Real.
        float3 trans_dir_pdf = 1;
        float3 trans_nee_pdf = 1;
        if (current_medium_id >= 0) {
            Medium medium = scene.media[current_medium_id];
            // In order to sample a distance, we adopt the null-scattering sampling strategy
            // from Galtier et al. "Radiative transfer and spectroscopic databases: A line-sampling Monte Carlo approach"
            // The idea is like this: we have the following radiative transfer equation:
            // L' = -sigma_t L + sigma_a L_e + sigma_s \int_{S^2} p L dw
            // integrating over distance we have the usual volume rendering equation:
            // L = \int_{0}^{z} T [sigma_a L_e + sigma_s \int_{S^2} p L dw] dz'
            // where T = exp(-\int_{0}^{z'} sigma_t dz'')
            // The exponentiation of T makes it difficult to estimate this integral
            // using Monte Carlo when sigma_t is varying with z''.
            // To deal with this, we use a trick called "homogenization":
            // we modify the RTE as follows:
            // L' = -(sigma_t + sigma_n)L + (sigma_a L_e + sigma_s \int_{S^2} p L dw + sigma_n L)
            // Note that this is just subtracting sigma_n first and then adding it
            // However, if we set sigma_n = M - sigma_t where M is some constant
            // which is an upperbound of sigma_t (usually called "majorant"),
            // the rendering equation becomes:
            // L = \int_{0}^{z} T_m [sigma_a L_e + sigma_s \int_{S^2} p L dw + sigma_n L] dz' + T_m[0->z] sigma_t L_s
            // where T_m = exp(-\int_{0}^{z'} M dz'') has a simple close-form solution.
            // An alternative way to look at this is to look at an alternative volume rendering equation,
            // by simply integrating the both sides of the RTE:
            // L = \int -sigma_t L + sigma_a L_e + sigma_s \int_{S^2} p L dw
            // then we apply control variates to this integral to obtain the null-scattering RTE.

            // Given L = \int_{0}^{z} T_m [sigma_a L_e + sigma_s \int_{S^2} p L dw + sigma_n L] dz'
            // We can then importance sample T_m, and then randomly chooses which term we want to
            // estimate in the [] bracket (we need to do this, otherwise we will have exponential
            // path branching since there are 2 recursive Ls).

            // We choose between the three terms based on sigma_a, sigma_s and sigma_n.
            // Furthermore, since our volumes are not emissive, we know that L_e = 0,
            // so we only need to sample the sigma_s term and sigma_n term.
            // Therefore we combine the scatter event and absorption event into a "collision event"
            // the collision event happens at probability sigma_t / majorant

            // Now we face a problem: the sigma_a/sigma_s/majorant have different values
            // at different wavelengthes! The above derivation only applies when
            // they are scalars.
            // Therefore in addition to sampling a distance, we also need to sample
            // a color component.
            // We also want to perform multiple importance sampling among all channels.
            // Thus we need to keep track of all PDFs in our stochastic evaluation.
            // This is discussed in
            // "A null-scattering path integral formulation of light transport"
            // from Miller et al.

            float max_t = float.maxValue;
            if (surface_vertex.hasValue) {
                max_t = distance(surface_vertex.value.position, ray.Origin);
            }
            float3 majorant = get_majorant(medium, ray);

            // Sample a channel for sampling
            float u = rng.next_float();
            int channel = clamp(int(u * float(3)), 0, 2);
            float ma_c = majorant[channel];
            float accum_t = 0;
            for (int count = 0; count < scene.options.max_null_collisions && ma_c > 0; count++) {
                float t = -log(1 - rng.next_float()) / ma_c;
                float dt = max_t - accum_t;
                // Update accumulated distance
                accum_t = min(accum_t + t, max_t);
                if (t < dt) { // Haven't reached a surface yet
                    float3 p = ray.Origin + accum_t * ray.Direction;
                    float3 sigma_a = get_sigma_a(medium, p);
                    float3 sigma_s = get_sigma_s(medium, p);
                    float3 sigma_t = sigma_s + sigma_a;
                    float3 sigma_n = majorant - sigma_t;
                    // With probability sigma_t / majorant, we sample a
                    // collision event and end the loop. Otherwise
                    // we continue sampling since we hit a null particle
                    // For numerical stability, we divide everything by the maximum of the majorant

                    float collision_prob = sigma_t[channel] / ma_c;
                    if (rng.next_float() <= collision_prob) {
                        scatter = true; // Tell the code to sample the phase function later
                        never_scatter = false;

                        // sigma_t will be accounted for later
                        transmittance *= exp(-majorant * t) / max_comp(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * sigma_t / max_comp(majorant);
                        current_path_throughput *= 1 / average(sigma_t);
                        // Update vertex information
                        vertex.position = ray.Origin + accum_t * ray.Direction;
                        vertex.interior_medium_id = current_medium_id;
                        vertex.exterior_medium_id = current_medium_id;
                        break;
                    }
                    else {
                        transmittance *= exp(-majorant * t) * sigma_n / max_comp(majorant);
                        trans_dir_pdf *= exp(-majorant * t) * sigma_n / max_comp(majorant);
                        trans_nee_pdf *= exp(-majorant * t) * majorant / max_comp(majorant);
                        current_path_throughput *= sigma_n / average(sigma_n);
                    }
                }
                else { // t >= max_t
                    // Only a single event when we reach the end
                    transmittance *= exp(-majorant * dt);
                    trans_dir_pdf *= exp(-majorant * dt);
                    trans_nee_pdf *= exp(-majorant * dt);
                    // No need to update vertex information: it's already correct
                    break;
                }
            }
        }
        // Update multiple bounces transmittion pdf.
        multi_trans_dir_pdf *= trans_dir_pdf;
        multi_trans_nee_pdf *= trans_nee_pdf;

        // Update current path throughput with transmittance
        //current_path_throughput *= (transmittance / average(trans_dir_pdf));
        NAN3_CHECK(current_path_throughput, "T[%v3f][%v3f]", transmittance, trans_dir_pdf);

        // If we didn't scatter and didn't reach a surface either,
        // we're done.
        if (!scatter && !surface_vertex.hasValue) {
            break;
        }

        // If we reach a surface and didn't scatter, include the emission.
        if (!scatter) {
            if (surface_vertex.hasValue) {
                if (vertex.area_light_id != -1) {
                    if (never_scatter) {
                        // This is the only way we can see the light source, so
                        // we don't need multiple importance sampling.
                        radiance += current_path_throughput *
                                    emission(vertex, -ray.Direction);
                        NAN3_CHECK(radiance, "H[%v3f]", current_path_throughput);
                    }
                    else {
                        // Need to account for next event estimation
                        int light_id = vertex.area_light_id;
                        // assert(light_id >= 0);
                        Light light = scene.lights[light_id];
                        PointAndNormal light_point = { vertex.position, vertex.geometric_normal };
                        // Note that p_nee needs to account for the path vertex that issued
                        // next event estimation potentially many bounces ago.
                        // The vertex position is stored in nee_p_cache.
                        // We also need to account for the ratio tracking pdf,
                        // which waas cached in multi_trans_nee_pdf
                        float p_nee = light_pmf(light_id) *
                                     pdf_point_on_light(light, light_point, nee_p_cache) *
                                     average(multi_trans_nee_pdf);
                        // The PDF for sampling the light source using directional sampling + transmittance sampling
                        // The directional sampling pdf was cached in dir_pdf in solid angle measure.
                        // The transmittance sampling pdf was cached in multi_trans_dir_pdf.
                        float3 light_dir = normalize(vertex.position - nee_p_cache);
                        float G = fabs(dot(vertex.geometric_normal, light_dir)) /
                                 distance_squared(nee_p_cache, vertex.position);
                        float p_dir = dir_pdf * average(multi_trans_dir_pdf) * G;
                        float w2 = 0.5;//(p_dir * p_dir) / (p_dir * p_dir + p_nee * p_nee);
                        // current_path_throughput already accounts for dir_pdf & transmittance.
                        radiance += current_path_throughput *
                                    emission(vertex, -ray.Direction) * w2;
                        NAN3_CHECK(radiance, "H[%v3f][%f] [%f][%f] [%v3f]", current_path_throughput, w2, p_dir, p_nee, multi_trans_dir_pdf);
                    }
                    return radiance;
                }
            }
        }

        if (max_depth != -1 && bounces == max_depth - 1) {
            break;
        }

        // If we reach a surface and it is index-matching, skip through it
        // if we hit nothing and we're not scattering, terminate.
        if (!scatter) {
            if (surface_vertex.value.material_id == -1) {
                ray = scene.make_ray(vertex.position, ray.Direction);
                current_medium_id =
                    update_medium_id(vertex, ray, current_medium_id);
                continue;
            }
        }

        // Cache the NEE vertex for later use
        nee_p_cache = vertex.position;
        // We have a scattering event (medium or surface), reset multi_trans_pdfs
        multi_trans_dir_pdf = 1;
        multi_trans_nee_pdf = 1;

        // next event estimation
        float3 C1 = 0;
        float w1 = 0;
        {
            // Sample a point on the light source.
            float light_rnd = rng.next_float();
            int light_id = sample_light(light_rnd);
            float3 point_rnd = rng.next_float3();
            Light light = scene.lights[light_id];
            PointAndNormal point_on_light = sample_point_on_light(light, vertex.position, point_rnd);
            // Compute transmittance to light. Skip through index-matching shapes.
            float3 T_light = 1;
            float3 C_light = 1;
            float3 p = vertex.position;
            int shadow_medium_id = current_medium_id;
            int shadow_bounces = 0;
            float3 p_trans_nee = 1;
            float3 p_trans_dir = 1;
            float3 p_pseudo = 1;
            while (true) {
                float3 dir_light = normalize(point_on_light.position - p);
                RayDesc shadow_ray;
                shadow_ray.Origin = p;
                shadow_ray.Direction = dir_light;
                shadow_ray.TMin = scene.shadow_epsilon();
				shadow_ray.TMax = (1 - scene.shadow_epsilon())
				                * distance(point_on_light.position, p);
                Optional<PathVertex> shadow_vertex = intersect(shadow_ray);
                float next_t = shadow_ray.TMax;
                if (shadow_vertex.hasValue) {
                    next_t = distance(p, shadow_vertex.value.position);
                }

                // Account for the transmittance to next_t
                // Unlike previous implementations, here we use "ratio tracking"
                // for estimating the transmittance. The idea is to
                // estimate exp(-\int_{0}^{max_t} sigma_t)
                // by sampling a distance t according to the majorant,
                // then weigh the samples using (majorant - sigma_t) / majorant
                // if t < max_t, otherwise weight them with sigma_t / majorant.
                // For numerical stability, we divide everything by the maximum of the majorant
                if (shadow_medium_id >= 0) {
                    Medium medium = scene.media[shadow_medium_id];
                    float3 majorant = get_majorant(medium, shadow_ray);

                    // Sample a channel for sampling
                    float u = rng.next_float();
                    int channel = clamp(int(u * float(3)), 0, 2);
                    float ma_c = majorant[channel];
                    float accum_t = 0;
                    for (int count = 0; count < scene.options.max_null_collisions && ma_c > 0; count++) {
                        float t = -log(1 - rng.next_float()) / ma_c;
                        float dt = next_t - accum_t;
                        accum_t = min(accum_t + t, next_t);
                        float3 p = shadow_ray.Origin + accum_t * shadow_ray.Direction;
                        float3 sigma_a = get_sigma_a(medium, p);
                        float3 sigma_s = get_sigma_s(medium, p);
                        float3 sigma_t = sigma_s + sigma_a;
                        float3 sigma_n = majorant - sigma_t;
                        if (t < dt) {
                            // didn't hit the surface, so this is a null-scattering event
                            T_light     *= exp(-majorant * t) * sigma_n / max_comp(majorant);
                            p_trans_nee *= exp(-majorant * t) * majorant / max_comp(majorant);
                            p_trans_dir *= exp(-majorant * t) * sigma_n / max_comp(majorant);
                            C_light     *= sigma_n / average(majorant);
							p_pseudo    *= majorant / sigma_n;
                            if (max_comp(T_light) <= 0) {
                                break;
                            }
                        }
                        else { // t >= next_t: we hit a surface
                            // account for the probability of sampling more than next_t
                            T_light     *= exp(-majorant * dt);
                            p_trans_nee *= exp(-majorant * dt);
                            p_trans_dir *= exp(-majorant * dt);
                            break;
                        }
                    }
                }

                if (!shadow_vertex.hasValue) {
                    // Nothing is blocking, we're done
                    break;
                }
                else {
                    // Something is blocking: is it an opaque surface?
                    if (shadow_vertex.value.material_id >= 0) {
                        // we're blocked
                        T_light = 0;
                        C_light = 0;
                        break;
                    }
                    // otherwise, we want to pass through -- this introduces
                    // one extra connection vertex
                    shadow_bounces++;
                    if (max_depth != -1 && bounces + shadow_bounces + 1 >= max_depth) {
                        // Reach the max no. of vertices
                        T_light = 0;
                        C_light = 0;
                        break;
                    }
                    // let's update and continue
                    shadow_medium_id = update_medium_id(shadow_vertex.value, shadow_ray, shadow_medium_id);
                    p = p + next_t * dir_light;
                }
            }

            if (max_comp(C_light) > 0) {
                // Compute sigma_s * T * T_light * G * f * L
                float3 dir_light = normalize(point_on_light.position - vertex.position);
                float G = max(-dot(dir_light, point_on_light.normal), float(0)) /
                         distance_squared(point_on_light.position, vertex.position);
                float p1 = light_pmf(light_id) *
                           pdf_point_on_light(light, point_on_light, vertex.position);
                float3 dir_view = -ray.Direction;
                float3 f;
                // are we on a surface or are we in a medium?
                if (scatter) {
                    // assert(current_medium_id >= 0);
                    Medium medium = scene.media[current_medium_id];
                    PhaseFunction phase_function = medium.phase_function;
                    f = eval(phase_function, dir_view, dir_light);
                }
                else {
                    Material mat = scene.materials[vertex.material_id];
                    f = eval(mat, vertex, dir_view, dir_light);
                }
                float3 sigma_s = 1;
                if (scatter) {
                    Medium medium = scene.media[current_medium_id];
                    sigma_s = get_sigma_s(medium, vertex.position);
                }
                float3 L = emission(light, -dir_light, 0, point_on_light);
                C1 = current_path_throughput * sigma_s * G * f * L * C_light / p1;
                // Multiple importance sampling: it's also possible
                // that a phase function sampling + multiple steps exponential sampling
                // will reach the light source.
                float p2 = 0;
                if (scatter) {
                    // assert(current_medium_id >= 0);
                    Medium medium = scene.media[current_medium_id];
                    p2 = pdf_sample_phase(medium.phase_function, dir_view, dir_light) * G;
                }
                else {
                    // assert(vertex.material_id >= 0);
                    Material mat = scene.materials[vertex.material_id];
                    p2 = pdf_sample_bsdf(mat, vertex, dir_view, dir_light) * G;
                }
                // p1 *= average(p_trans_nee);
                // p2 *= average(p_trans_dir);
                w1 = 0.5;//((p1 * p1) / (p1 * p1 + p2 * p2)) * average(p_pseudo);
                //NAN3_CHECK(w1, "N[%v3f][%v3f][%f][%f]", p_trans_nee, p_trans_dir, p1, p2);
            }
        }
        radiance += C1 * w1;
        //NAN3_CHECK(radiance, "N[%v3f][%f]", C1, w1);

        // Sample the next direction & update current_path_throughput
        float3 next_dir;
        if (scatter) {
            // Phase function sampling
            // assert(current_medium_id >= 0);
            Medium medium = scene.media[current_medium_id];
            float3 dir_view = -ray.Direction;
            float2 phase_rnd_param = rng.next_float2();
            Optional<float3> phase_sample =
                sample_phase_function(medium.phase_function, dir_view,
                                      phase_rnd_param);
            if (!phase_sample.hasValue) {
                // Phase function sampling failed. Abort the loop.
                break;
            }
            next_dir = phase_sample.value;
            float3 f = eval(medium.phase_function, dir_view, next_dir);
            float p2 = pdf_sample_phase(medium.phase_function, dir_view, next_dir);
            dir_pdf = p2;
            // Need to remember multiplying the scattering coefficient!
            float3 sigma_s = get_sigma_s(medium, vertex.position);
            current_path_throughput *= sigma_s * (f / p2);
            NAN3_CHECK(current_path_throughput, "S[%v3f][%v3f][%f]", sigma_s, f, p2);
        }
        else {
            // BSDF sampling
            Material mat = scene.materials[vertex.material_id];
            float3 dir_view = -ray.Direction;
            float3 bsdf_rnd_param = rng.next_float3();
            Optional<BSDFSampleRecord> bsdf_sample =
                sample_bsdf(mat, vertex, dir_view, bsdf_rnd_param);
            if (!bsdf_sample.hasValue) {
                // Phase function sampling failed. Abort the loop.
                break;
            }
            next_dir = bsdf_sample.value.dir_out;
            float3 f = eval(mat, vertex, dir_view, next_dir);
            float p2 = pdf_sample_bsdf(mat, vertex, dir_view, next_dir);
            dir_pdf = p2;
            current_path_throughput *= (f / p2);
            NAN3_CHECK(current_path_throughput, "S[%v3f][%f]", f, p2);
        }
        never_scatter = false;

        // Update rays/current_path_throughput/current_pdf
        // Russian roulette heuristics
        float rr_prob = 1;
        if (bounces >= scene.options.rr_depth) {
            rr_prob = min(max_comp(current_path_throughput), 0.95);
            if (rng.next_float() > rr_prob) {
                // Terminate the path
                break;
            }
            current_path_throughput /= rr_prob;
        }

        // Update rays
        ray = scene.make_ray(vertex.position, next_dir);
        current_medium_id =
            update_medium_id(vertex, ray, current_medium_id);
    }
    return radiance;
}
