#include "../microfacet.h"

constexpr auto R_0 (Real n) -> Real
{
    return P2(n - Real(1)) / P2(n + Real(1));
}

static Spectrum disney_clearcoat(
    Frame    frame,
    Real     clearcoat_gloss,
    Vector3  half_vector,
    Vector3  dir_in,
    Vector3  dir_out
)
{
    Real alpha_g = (Real(1) - clearcoat_gloss) * Real(0.1) + clearcoat_gloss * Real(0.001);
    
    Real F_c = R_0(1.5) + (Real(1) - R_0(1.5)) * P5(Real(1) - abs(dot(half_vector, dir_out)));
    Real hl_z = dot(half_vector, frame.n);
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

static Real disney_clearcoat_pdf(
    PathVertex const& vertex,
    Vector3 dir_in,
    Vector3 dir_out,
    Real clearcoat_gloss
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

    Vector3 h = normalize(dir_in + dir_out);
    Real n_dot_in = dot(frame.n, dir_in);
    //Real n_dot_out = dot(frame.n, dir_out);
    Real n_dot_h = dot(frame.n, h);
    
    Real alpha_g = (Real(1) - clearcoat_gloss) * Real(0.1) + clearcoat_gloss * Real(0.001);
    
    Real hl_z = dot(h, frame.n);
    Real D_c = (P2(alpha_g) - Real(1))
             / (c_PI * log(P2(alpha_g)) * (Real(1) + (P2(alpha_g) - Real(1)) * P2(hl_z)));

    // (4 * cos_theta_v) is the Jacobian of the reflectiokn
    return (D_c * n_dot_h) / (4 * n_dot_in); 
}

static BSDFSampleRecord disney_clearcoat_sample(
    PathVertex const& vertex,
    Vector3 dir_in,
    Vector2 rnd_param_uv,
    Real clearcoat_gloss
)
{
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
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

    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Vector3 half_vector = normalize(dir_in + dir_out);
    return disney_clearcoat(frame, clearcoat_gloss, half_vector, dir_in, dir_out);
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    return disney_clearcoat_pdf(vertex, dir_in, dir_out, clearcoat_gloss);
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    return disney_clearcoat_sample(vertex, dir_in, rnd_param_uv, clearcoat_gloss);
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
