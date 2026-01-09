#include "../microfacet.h"

constexpr auto R_0 (Real n) -> Real
{
    return P2(n - Real(1)) / P2(n + Real(1));
}

Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const {
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

    // Half-Vector
    Vector3 h = normalize(dir_in + dir_out);
    
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (Real(1) - clearcoat_gloss) * Real(0.1) + clearcoat_gloss * Real(0.001);
    
    Real F_c = R_0(1.5) + (Real(1) - R_0(1.5)) * P5(Real(1) - abs(dot(h, dir_out)));
    Real hl_z = dot(h, frame.n);
    Real D_c = (P2(alpha_g) - Real(1))
             / (c_PI * log(P2(alpha_g)) * (Real(1) + (P2(alpha_g) - Real(1)) * P2(hl_z)));

    auto G = [&](Vector3 w) {
        Real wl_x = dot(w, frame.x);
        Real wl_y = dot(w, frame.y);
        Real wl_z = dot(w, frame.n);
        Real x = (P2(wl_x * Real(0.25)) + P2(wl_y * Real(0.25))) / P2(wl_z);
        Real V = Real(0.5) * (sqrt(Real(1) + x) - Real(1));
        return Real(1) / (Real(1) + V);
    };
    Real G_c = G(dir_in) * G(dir_out);

    return make_const_spectrum(F_c * D_c * G_c / (Real(4) * abs(dot(frame.n, dir_in))));
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
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

    Vector3 h = normalize(dir_in + dir_out);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    Real n_dot_h = dot(frame.n, h);
    
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (Real(1) - clearcoat_gloss) * Real(0.1) + clearcoat_gloss * Real(0.001);
    
    Real hl_z = dot(h, frame.n);
    Real D_c = (P2(alpha_g) - Real(1))
             / (c_PI * log(P2(alpha_g)) * (Real(1) + (P2(alpha_g) - Real(1)) * P2(hl_z)));

    // (4 * cos_theta_v) is the Jacobian of the reflectiokn
    return (D_c * n_dot_h) / (4 * n_dot_in); 
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // Convert the incoming direction to local coordinates
    Vector3 local_dir_in = to_local(frame, dir_in);
    
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha = (Real(1) - clearcoat_gloss) * Real(0.1) + clearcoat_gloss * Real(0.001);
    
    Vector3 local_micro_normal =
        sample_half_vector(alpha, rnd_param_uv);
    
    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, sqrt(alpha) /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
