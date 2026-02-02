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

// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    Ray ray = sample_primary(scene.camera, x, y, rng);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, {});
    if (!vertex_) {
        return background_emission(scene, ray, {});
    }
    PathVertex& vertex = *vertex_;

    const Medium &medium = scene.media[vertex.exterior_medium_id];

    Spectrum transmittance = exp(-get_sigma_a(medium, vertex.position) * distance(ray.org, vertex.position));

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
    bool hit = false;
    std::optional<PathVertex> vertex_ = intersect(scene, ray, {});
    if (vertex_) {
        hit_t = distance(ray.org, vertex_->position);
        hit = true;
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

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
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
