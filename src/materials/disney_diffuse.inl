static Spectrum disney_diffuse(
    Frame    frame,
    Spectrum base_color,
    Real     roughness,
    Real     subsurface,
    Vector3  half_vector,
    Vector3  dir_in,
    Vector3  dir_out
)
{
    auto F_D = [&](Vector3 w) -> Real {
        auto F_D90 = Real(0.5) + Real(2) * roughness * P2(dot(half_vector, dir_out));
        return Real(1) + (F_D90 - Real(1)) * P5(Real(1) - abs(dot(frame.n, w)));
    };
    Spectrum base_diffuse = (base_color / c_PI) * F_D(dir_in) * F_D(dir_out) * abs(dot(frame.n, dir_out));

    auto F_SS = [&](Vector3 w) -> Real {
        auto F_SS90 = roughness * P2(dot(half_vector, dir_out));
        return Real(1) + (F_SS90 - Real(1)) * P5(Real(1) - abs(dot(frame.n, w)));
    };

    Real a = Real(1)
           / (abs(dot(frame.n, dir_in)) + abs(dot(frame.n, dir_out))) - Real(0.5);
    Spectrum subsurface_color = (Real(1.25) / c_PI) * base_color
                        * (F_SS(dir_in) * F_SS(dir_out) * a + Real(0.5))
                        * abs(dot(frame.n, dir_out));

    return base_diffuse * (Real(1) - subsurface)
         + subsurface * subsurface_color;
}

static Real disney_diffuse_pdf(
    PathVertex const& vertex,
    Vector3 dir_in,
    Vector3 dir_out
)
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
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

static BSDFSampleRecord disney_diffuse_sample(
    PathVertex const& vertex,
    Vector3 dir_in,
    Vector2 rnd_param_uv
)
{
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        //assert(false && "DIFFUSE");
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
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Vector3 half_vector = normalize(dir_in + dir_out);
    return disney_diffuse(frame, base_color, roughness, subsurface, half_vector, dir_in, dir_out);
}

Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    return disney_diffuse_pdf(vertex, dir_in, dir_out);
}

std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
    return disney_diffuse_sample(vertex, dir_in, rnd_param_uv);
}

TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
