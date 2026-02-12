#pragma once

Ray sample_primary(const Camera &camera, int x, int y, pcg32_state &rng) {
    int w = camera.width, h = camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    return sample_primary(camera, screen_pos);
}

Spectrum background_emission(const Scene &scene, Ray ray, RayDifferential ray_diff) {
    // Hit background. Account for the environment map if needed.
    if (has_envmap(scene)) {
        const Light &envmap = get_envmap(scene);
        return emission(envmap,
                        -ray.dir, // pointing outwards from light
                        ray_diff.spread,
                        PointAndNormal{}, // dummy parameter for envmap
                        scene);
    }
    return make_zero_spectrum();
}

std::tuple<Spectrum, Real> L_s1(
    const Scene &scene,
    const Vector3 &dir_in,
    const Vector3 &dir_out,
    const Medium &medium,
    const Vector3 &p,
    const PointAndNormal &point_on_light,
    Real sigma_t
) {
    PhaseFunction phase_function = get_phase_function(medium);
    Real dist_sq = length_squared(point_on_light.position - p);
    Real t_prime = sqrt(dist_sq);
    Spectrum rho = eval(phase_function, dir_in, dir_out);

    return {
        rho * exp(-sigma_t * t_prime) * (abs(dot(dir_out, point_on_light.normal)) / dist_sq),
        1.0,
    };
}

auto get_sigma_t(const Medium &medium, const Vector3 &p) {
    return get_sigma_a(medium, p) + get_sigma_s(medium, p);
}

// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    Ray ray = sample_primary(scene.camera, x, y, rng);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, {});
    if (!vertex_) {
        return make_const_spectrum(0);
    }
    PathVertex& vertex = *vertex_;

    const Medium &medium = scene.media[scene.camera.medium_id];

    Real t = distance(ray.org, vertex.position);
    Real sigma_a = get_sigma_a(medium, {}).x;
    Real transmittance = exp(-sigma_a * t);

    Spectrum Le = make_zero_spectrum();
    if (is_light(scene.shapes[vertex.shape_id])) {
        Le = emission(vertex, -ray.dir, scene);
    }
    return transmittance * Le;
}

// The second simplest volumetric renderer: 
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    
    Ray ray = sample_primary(scene.camera, x, y, rng);

    // Assumption: There is only a single, homogeneous (σa and σs are constants over space) volume.
    assert(scene.camera.medium_id != -1);
    const Medium &medium = scene.media[scene.camera.medium_id];
    // Assumption: The volume is monochromatic: the three color channels of σs and σa have the same values.
    Real sigma_s = get_sigma_s(medium, {}).x;
    Real sigma_t = get_sigma_a(medium, {}).x + sigma_s;

    Real u = next_pcg32_real<Real>(rng); // [0, 1]
    Real t = -log(Real(1) - u) / sigma_t;

    Real hit_t = infinity<Real>();
    std::optional<PathVertex> vertex_ = intersect(scene, ray, {});
    if (vertex_) {
        hit_t = distance(ray.org, vertex_->position);
    }

    if (t < hit_t) {
        Real trans_pdf = exp(-sigma_t * t) * sigma_t;
        Real transmittance = exp(-sigma_t * t);
        Vector3 p = ray.org + t * ray.dir;

        // First, we sample a point on the light source.
        // We do this by first picking a light source, then pick a point on it.
        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        PointAndNormal point_on_light = sample_point_on_light(light, p, light_uv, shape_w, scene);
        // Throughout the homework, we assume there is no environment map in the scene.
        Vector3 dir_light = normalize(point_on_light.position - p);
        Spectrum Le = emission(light, -dir_light, 0, point_on_light, scene);
        Real pdf_light = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, p, scene);

        auto [L_s1_estimate, L_s1_pdf] = L_s1(scene, -ray.dir, dir_light, medium, p, point_on_light, sigma_t);
        return (transmittance / trans_pdf) * sigma_s * (L_s1_estimate * Le / (L_s1_pdf * pdf_light));
    }
    else {
        // hit a surface, account for surface emission
        Real trans_pdf = exp(-sigma_t * hit_t);
        Real transmittance = exp(-sigma_t * hit_t);

        Spectrum Le = make_zero_spectrum();
        if (is_light(scene.shapes[vertex_.value().shape_id])) {
            Le = emission(*vertex_, -ray.dir, scene);
        }
        return (transmittance / trans_pdf) * Le;
    }
}

static int update_medium(Ray ray, PathVertex vertex, int medium_id)
{
    if (vertex.interior_medium_id != vertex.exterior_medium_id)
        return dot(ray.dir, vertex.geometric_normal) > 0 ?
            vertex.exterior_medium_id : vertex.interior_medium_id;
    return medium_id;
}

static Spectrum Le(const Scene &scene, const Ray &ray, const PathVertex &vertex)
{
    if (is_light(scene.shapes[vertex.shape_id]))
        return emission(vertex, -ray.dir, scene);
    return make_zero_spectrum();
}

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    
    Ray ray = sample_primary(scene.camera, x, y, rng);
    int current_medium_id = scene.camera.medium_id;

    Real current_path_throughput = Real(1);
    Spectrum radiance = make_const_spectrum(0);
    int bounces = 0;

    while (true)
    {
        bool scatter = false;

        Real t_hit = infinity<Real>();
        std::optional<PathVertex> vertex_ = intersect(scene, ray, {});
        if (vertex_)
            t_hit = distance(ray.org, vertex_->position);
        
        Real transmittance = Real(1);
        Real trans_pdf = Real(1);
        if (current_medium_id >= 0)
        {
            const Medium &medium = scene.media[current_medium_id];

            // Sample t s.t. p(t) ~ exp(-sigma_t * t)
            Real sigma_s = get_sigma_s(medium, {}).x;
            Real sigma_t = get_sigma_a(medium, {}).x + sigma_s;
            Real u = next_pcg32_real<Real>(rng); // [0, 1]
            Real t = -log(Real(1) - u) / sigma_t;

            // Compute transmittance and trans_pdf
            if (t < t_hit)
            {
                trans_pdf = exp(-sigma_t * t) * sigma_t;
                transmittance = exp(-sigma_t * t);
                scatter = true;
            }
            else
            {
                t = t_hit;
                trans_pdf = exp(-sigma_t * t);
                transmittance = exp(-sigma_t * t);
            }

            ray.org = ray.org + t * ray.dir;
        }

        current_path_throughput *= (transmittance / trans_pdf);

        if (!scatter && vertex_)
            radiance += current_path_throughput * Le(scene, ray, *vertex_);

        if (bounces == scene.options.max_depth - 1
         && scene.options.max_depth != -1)
            break;
        
        if (!scatter && vertex_)
        {
            PathVertex &vertex = *vertex_;
            if (vertex.material_id == -1)
            {
                current_medium_id = update_medium(ray, vertex, current_medium_id);
                bounces += 1;
                continue;
            }
        }

        if (scatter)
        {
            const Medium &medium = scene.media[current_medium_id];
            Real sigma_s = get_sigma_s(medium, {}).x;
            PhaseFunction phase_function = get_phase_function(medium);
            Vector2 rnd_param_uv = { next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            Vector3 next_dir = *sample_phase_function(phase_function, -ray.dir, rnd_param_uv);
            current_path_throughput *= (eval(phase_function, -ray.dir, next_dir).x
                                      / pdf_sample_phase(phase_function, -ray.dir, next_dir)) * sigma_s;
            
            // Update ray direction
            ray.dir = next_dir;
        }
        else break; // Hit a surface ....

        Real rr_prob = Real(1);
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = min(current_path_throughput, Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob)
                break;
            else
                current_path_throughput /= rr_prob;
        }
        bounces += 1;
    }

    return radiance;
}

static Spectrum next_event_estimation(const Scene &scene,
    const Material &mat, const PathVertex &vertex,
    pcg32_state& rng, const Ray& ray, int current_medium_id,
    int bounces, bool use_phase_pdf = true)
{
    // Sample point on light
    // First, we sample a point on the light source.
    // We do this by first picking a light source, then pick a point on it.
    Vector2 light_uv { next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    assert(light_id >= 0 && "Could not sample light for NEE!");
    const Light &light = scene.lights[light_id];
    PointAndNormal point_on_light = sample_point_on_light(light, ray.org, light_uv, shape_w, scene);
    // Throughout the homework, we assume there is no environment map in the scene.
    Vector3 dir_light = normalize(point_on_light.position - ray.org);
    Real pdf_nee = light_pmf(scene, light_id) * pdf_point_on_light(light, point_on_light, ray.org, scene);
    
    Spectrum Le = emission(light, -dir_light, Real(0), point_on_light, scene);

    Vector3 p = ray.org;
    Vector3 p_prime = point_on_light.position;

    // Compute transmittance to light. Skip through index-matching shapes.
    Real T_light = Real(1);
    int shadow_medium_id = current_medium_id;
    int shadow_bounces = 0;
    Real p_trans_dir = Real(1); // for multiple importance sampling

    while (true)
    {
        Ray shadow_ray = { p, dir_light, 
                               get_shadow_epsilon(scene),
                               (1 - get_shadow_epsilon(scene)) *
                                   distance(point_on_light.position, p)};
        std::optional<PathVertex> isect = intersect(scene, shadow_ray);
        Real next_t = distance(p, p_prime);
        if (isect)
            next_t = distance(p, isect->position);
        
        // Account for the transmittance to next_t
        if (shadow_medium_id != -1)
        {
            const Medium& medium = scene.media[shadow_medium_id];
            Real sigma_t = get_sigma_t(medium, {}).x;
            T_light *= exp(-sigma_t * next_t);
            p_trans_dir *= exp(-sigma_t * next_t);
        }

        // Nothing is blocking, we're done
        if (!isect)
            break;

        PathVertex& vertex = *isect;

        // Something is blocking: is it an opaque surface?
        if (vertex.material_id >= 0)
            return make_const_spectrum(0); // we're blocked

        // otherwise, it's an index-matching surface and
        // we want to pass through -- this introduces
        // one extra connection vertex
        shadow_bounces += 1;
        if (scene.options.max_depth != -1 && bounces + shadow_bounces + 1 >= scene.options.max_depth)
            return make_const_spectrum(0);

        shadow_medium_id = update_medium(shadow_ray, vertex, shadow_medium_id);
        p = p + next_t * dir_light;
    }

    if (T_light > 0)
    {
        Real G = abs(dot(-dir_light, point_on_light.normal)) / length_squared(ray.org - p_prime);

        Spectrum f = make_const_spectrum(0);
        Real pdf_other = 0;
        if (use_phase_pdf)
        {
            const Medium& medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);
            f = eval(phase_function, -ray.dir, dir_light);
            // Multiple importance sampling: it's also possible
            // that a phase function sampling + multiple exponential sampling
            // will reach the light source.
            // We also need to multiply with G to convert phase function PDF to area measure.
            pdf_other = pdf_sample_phase(phase_function, -ray.dir, dir_light) * G * p_trans_dir;
        } else {
            f = eval(mat, -ray.dir, dir_light, vertex, scene.texture_pool);
            pdf_other = pdf_sample_bsdf(mat, -ray.dir, dir_light, vertex, scene.texture_pool) * G;
        }

        Spectrum contrib = T_light * G * f / pdf_nee;
        
        // power heuristics
        Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_other * pdf_other);
        return Le * w * contrib;
    }

    return make_const_spectrum(0);
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    Ray ray = sample_primary(scene.camera, x, y, rng);
    int current_medium_id = scene.camera.medium_id;

    bool never_scatter = true;
    Real current_path_throughput = Real(1);
    Spectrum radiance = make_const_spectrum(0);
    int bounces = 0;
    Real dir_pdf = 0; // in solid angle measure
    Vector3 nee_p_cache;
    Real multi_trans_pdf = Real(1);

    while (true)
    {
        bool scatter = false;

        Real t_hit = infinity<Real>();
        std::optional<PathVertex> vertex_ = intersect(scene, ray);
        if (vertex_)
            t_hit = distance(ray.org, vertex_->position);
        
        Real t_next = t_hit;
        Real transmittance = Real(1);
        Real trans_pdf = Real(1);
        if (current_medium_id >= 0)
        {
            const Medium &medium = scene.media[current_medium_id];

            // Sample t s.t. p(t) ~ exp(-sigma_t * t)
            Real sigma_s = get_sigma_s(medium, {}).x;
            Real sigma_t = get_sigma_a(medium, {}).x + sigma_s;
            Real u = next_pcg32_real<Real>(rng); // [0, 1]
            Real t = -log(Real(1) - u) / sigma_t;

            // Compute transmittance and trans_pdf
            if (t < t_hit)
            {
                trans_pdf = exp(-sigma_t * t) * sigma_t;
                transmittance = exp(-sigma_t * t);
                scatter = true;
            }
            else
            {
                t = t_hit;
                trans_pdf = exp(-sigma_t * t);
                transmittance = exp(-sigma_t * t);
            }

            t_next = t;
        }
        ray.org = ray.org + t_next * ray.dir;
        current_path_throughput *= (transmittance / trans_pdf);
		multi_trans_pdf *= trans_pdf;

        if (!scatter)
        {
            if (never_scatter)
            {
				if (vertex_)
                	radiance += current_path_throughput * Le(scene, ray, *vertex_);
            }
            else if (vertex_ && is_light(scene.shapes[vertex_->shape_id]))
            {
                PathVertex& vertex = *vertex_;

                // Need to account for next event estimation
                PointAndNormal light_point { vertex.position, vertex.geometric_normal };
                // Note that pdf_nee needs to account for the path vertex that issued
                // next event estimation potentially many bounces ago.
                // The vertex position is stored in nee_p_cache.
                int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                assert(light_id >= 0);
                const Light &light = scene.lights[light_id];
                Real pdf_nee = light_pmf(scene, light_id) * pdf_point_on_light(light, light_point, nee_p_cache, scene);
                // The PDF for sampling the light source using phase function sampling + transmittance sampling
                // The directional sampling pdf was cached in dir_pdf in solid angle measure.
                // The transmittance sampling pdf was cached in multi_trans_pdf.
                Vector3 dir_light = normalize(light_point.position - nee_p_cache);
                Real G = abs(dot(-dir_light, light_point.normal)) / length_squared(nee_p_cache - light_point.position);
                Real dir_pdf_ = dir_pdf * multi_trans_pdf * G;
				//printf("HOW %.6f = %.6f * %.6f * %.6f (%.6f)\n", dir_pdf_, dir_pdf, multi_trans_pdf, G, length_squared(nee_p_cache - light_point.position));
                Real w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);
                // current_path_throughput already accounts for transmittance.
                radiance += current_path_throughput * emission(vertex, -ray.dir, scene) * w;
            }
        }

        if (bounces == scene.options.max_depth - 1
         && scene.options.max_depth != -1)
            break;
        
        if (!scatter && vertex_)
        {
            PathVertex &vertex = *vertex_;
            if (vertex.material_id == -1)
            {
                current_medium_id = update_medium(ray, vertex, current_medium_id);
                bounces += 1;
                continue;
            }
        }

        if (scatter)
        {
            const Medium &medium = scene.media[current_medium_id];
            Real sigma_s = get_sigma_s(medium, {}).x;

			nee_p_cache = ray.org;
            radiance += current_path_throughput
                * next_event_estimation(scene, {}, {}, rng, ray, current_medium_id, bounces)
                * sigma_s;
            
            PhaseFunction phase_function = get_phase_function(medium);
            Vector2 rnd_param_uv = { next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            Vector3 next_dir = *sample_phase_function(phase_function, -ray.dir, rnd_param_uv);
            dir_pdf = pdf_sample_phase(phase_function, -ray.dir, next_dir);
			multi_trans_pdf = Real(1);
            current_path_throughput *= (eval(phase_function, -ray.dir, next_dir).x
                                      / dir_pdf)
                                    * sigma_s;

            // Update ray direction
            ray.dir = next_dir;
            never_scatter = false;
        }
        else break; // Hit a surface ....

        Real rr_prob = Real(1);
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = min(current_path_throughput, Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob)
                break;
            else
                current_path_throughput /= rr_prob;
        }
        bounces += 1;
    }

    return radiance;
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    Ray ray = sample_primary(scene.camera, x, y, rng);
    int current_medium_id = scene.camera.medium_id;

    bool never_mis = true;
    bool phase_sampling = false;
    Spectrum current_path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_const_spectrum(0);
    int bounces = 0;
    Real dir_pdf = 0; // in solid angle measure
    Real bsdf_pdf = 0; // in solid angle measure
    Vector3 nee_p_cache;
    Real multi_trans_pdf = Real(1);
    Real eta_scale = Real(1);

    while (true)
    {
        bool scatter = false;

        Real t_hit = infinity<Real>();
        std::optional<PathVertex> vertex_ = intersect(scene, ray);
        if (vertex_)
            t_hit = distance(ray.org, vertex_->position);
        
        Real t_next = t_hit;
        Real transmittance = Real(1);
        Real trans_pdf = Real(1);
        if (current_medium_id >= 0)
        {
            const Medium &medium = scene.media[current_medium_id];

            // Sample t s.t. p(t) ~ exp(-sigma_t * t)
            Spectrum sigma_s = get_sigma_s(medium, {});
            Spectrum sigma_t = get_sigma_a(medium, {}) + sigma_s;
            Real u = next_pcg32_real<Real>(rng); // [0, 1]
            Real t = -log(Real(1) - u) / sigma_t.x;

            // Compute transmittance and trans_pdf
            if (t < t_hit)
            {
                trans_pdf = exp(-sigma_t.x * t) * sigma_t.x;
                transmittance = exp(-sigma_t.x * t);
                scatter = true;
            }
            else
            {
                t = t_hit;
                trans_pdf = exp(-sigma_t.x * t);
                transmittance = exp(-sigma_t.x * t);
            }

            t_next = t;
        }
        ray.org = ray.org + t_next * ray.dir;
        current_path_throughput *= (transmittance / trans_pdf);
		multi_trans_pdf *= trans_pdf;

        if (!scatter)
        {
            if (never_mis)
            {
				if (vertex_)
                	radiance += current_path_throughput * Le(scene, ray, *vertex_);
            }
            else if (vertex_ && is_light(scene.shapes[vertex_->shape_id]))
            {
                PathVertex& vertex = *vertex_;

                // Need to account for next event estimation
                PointAndNormal light_point { vertex.position, vertex.geometric_normal };
                // Note that pdf_nee needs to account for the path vertex that issued
                // next event estimation potentially many bounces ago.
                // The vertex position is stored in nee_p_cache.
                int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                assert(light_id >= 0);
                const Light &light = scene.lights[light_id];
                Real pdf_nee = light_pmf(scene, light_id) * pdf_point_on_light(light, light_point, nee_p_cache, scene);
                // The PDF for sampling the light source using phase function sampling + transmittance sampling
                // The directional sampling pdf was cached in dir_pdf in solid angle measure.
                // The transmittance sampling pdf was cached in multi_trans_pdf.
                Vector3 dir_light = normalize(light_point.position - nee_p_cache);
                Real G = fabs(dot(-dir_light, light_point.normal)) / length_squared(nee_p_cache - light_point.position);

                Real w;
                if (phase_sampling) {
                    Real dir_pdf_ = dir_pdf * multi_trans_pdf * G;
                    w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + pdf_nee * pdf_nee);
                } else {
                    Real bsdf_pdf_ = bsdf_pdf * G;
                    w = (bsdf_pdf_ * bsdf_pdf_) / (bsdf_pdf_ * bsdf_pdf_ + pdf_nee * pdf_nee);
                }
                // current_path_throughput already accounts for transmittance.
                radiance += current_path_throughput * emission(vertex, -ray.dir, scene) * w;
            }
        }

        if (bounces == scene.options.max_depth - 1
         && scene.options.max_depth != -1)
            break;
        
        if (!scatter && vertex_)
        {
            PathVertex &vertex = *vertex_;
            if (vertex.material_id == -1)
            {
                current_medium_id = update_medium(ray, vertex, current_medium_id);
                bounces += 1;
                continue;
            }
        }

        if (scatter)
        {
            const Medium &medium = scene.media[current_medium_id];
            Spectrum sigma_s = get_sigma_s(medium, {});

			nee_p_cache = ray.org;
            radiance += current_path_throughput
                * next_event_estimation(scene, {}, {}, rng, ray, current_medium_id, bounces)
                * sigma_s;
            
            PhaseFunction phase_function = get_phase_function(medium);
            Vector2 rnd_param_uv = { next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng) };
            Vector3 next_dir = *sample_phase_function(phase_function, -ray.dir, rnd_param_uv);
            dir_pdf = pdf_sample_phase(phase_function, -ray.dir, next_dir);
			multi_trans_pdf = Real(1);
            current_path_throughput *= (eval(phase_function, -ray.dir, next_dir)
                                      / dir_pdf)
                                    * sigma_s;

            // Update ray direction
            ray.dir = next_dir;
            never_mis = false;
            phase_sampling = true;
        }
        else if (vertex_) {
            const PathVertex &vertex = *vertex_;
            const Material &mat = scene.materials[vertex.material_id];

			nee_p_cache = ray.org;
            radiance += current_path_throughput
                * next_event_estimation(scene, mat, vertex, rng, ray, current_medium_id, bounces, false);

            // Let's do the hemispherical sampling next.
            Vector3 dir_view = -ray.dir;
            Vector2 bsdf_rnd_param_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
            std::optional<BSDFSampleRecord> bsdf_sample_ =
                sample_bsdf(mat,
                            dir_view,
                            vertex,
                            scene.texture_pool,
                            bsdf_rnd_param_uv,
                            bsdf_rnd_param_w);
            if (!bsdf_sample_) {
                // BSDF sampling failed. Abort the loop.
                break;
            }
            const BSDFSampleRecord &bsdf_sample = *bsdf_sample_;
            Vector3 dir_bsdf = bsdf_sample.dir_out;
            // Update eta_scale
            if (bsdf_sample.eta == 0); else {
                eta_scale /= (bsdf_sample.eta * bsdf_sample.eta);
            }

            Spectrum f = eval(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            bsdf_pdf = pdf_sample_bsdf(mat, dir_view, dir_bsdf, vertex, scene.texture_pool);
            if (bsdf_pdf <= 0) {
                // Numerical issue -- we generated some invalid rays.
                break;
            }
            current_path_throughput *= f / bsdf_pdf;

            // Update ray direction
            ray.org = vertex.position;
            ray.dir = dir_bsdf;
            ray.tnear = get_intersection_epsilon(scene);
            ray.tfar = infinity<Real>();
            phase_sampling = false;
            never_mis = false;
        }

        Real rr_prob = Real(1);
        if (bounces >= scene.options.rr_depth)
        {
            rr_prob = min(max((1 / eta_scale) * current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob)
                break;
        }
        bounces += 1;
        current_path_throughput /= rr_prob;
    }

    return radiance;
}

// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}
