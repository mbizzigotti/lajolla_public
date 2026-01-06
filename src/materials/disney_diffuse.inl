Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface_weight = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);

    // Half-Vector
    Vector3 h = normalize(dir_in + dir_out);

    auto F_D = [&](Vector3 w) -> Real {
        auto F_D90 = Real(0.5) + Real(2) * roughness * P2(dot(h, dir_out));
        return 1 + (F_D90 - Real(1)) * (Real(1) - P5(abs(dot(frame.n, w))));
    };
    Spectrum base_diffuse = (base_color / c_PI) * F_D(dir_in) * F_D(dir_out) * abs(dot(frame.n, dir_out));

    auto F_SS = [&](Vector3 w) -> Real {
        auto F_SS90 = roughness * P2(dot(h, dir_out));
        return Real(1) + (F_SS90 - Real(1)) * (Real(1) - P5(abs(dot(frame.n, w))));
    };
    Real a = Real(1) / (abs(dot(frame.n, dir_in)) + abs(dot(frame.n, dir_out))) + Real(0.5);
    Spectrum subsurface = (Real(1.25) / c_PI) * base_color
                        * (F_SS(dir_in) * F_SS(dir_out) * a + Real(0.5))
                        * abs(dot(frame.n, dir_out));

    return base_diffuse * (Real(1) - subsurface_weight)
         + subsurface * subsurface_weight;
}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, Real(1) /* roughness */};
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
