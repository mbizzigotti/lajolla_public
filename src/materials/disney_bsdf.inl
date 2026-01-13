#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Vector3 half_vector = normalize(dir_in + dir_out);

    Spectrum f_diffuse   = make_zero_spectrum();
    Spectrum f_metal     = make_zero_spectrum();
    Spectrum f_clearcoat = make_zero_spectrum();
    Spectrum f_sheen     = make_zero_spectrum();

    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    if (dot(dir_in, vertex.geometric_normal) > 0) {
        Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real specular_tint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real eta = bsdf.eta;

        Real L = luminance(base_color);
        Spectrum C_tint = L > 0 ? base_color / L : make_const_spectrum(1);
        Spectrum K_s = (Real(1) - specular_tint) + specular_tint * C_tint;
        Spectrum C_0 = specular * R_0(eta) * (1 - metallic) * K_s + metallic * base_color;
        Spectrum F_m = C_0 + (Real(1) - C_0) * P5(Real(1) - dot(half_vector, dir_out));

        f_diffuse   = (Real(1) - specular_transmission) * (Real(1) - metallic)
                    * disney_diffuse(frame, base_color, roughness, subsurface, half_vector, dir_in, dir_out);
        f_sheen     = (Real(1) - metallic) * sheen
                    * disney_sheen(frame, base_color, sheen_tint, half_vector, dir_in, dir_out);
        f_metal     = (Real(1) - specular_transmission * (Real(1) - metallic))
                    * disney_metal(frame, F_m, roughness, anisotropic, half_vector, dir_in, dir_out);
        f_clearcoat = Real(0.25) * clearcoat
                    * disney_clearcoat(frame, clearcoat_gloss, half_vector, dir_in, dir_out);
    }

    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    Spectrum f_glass = (Real(1) - metallic) * specular_transmission
                     * disney_glass(frame, base_color, eta, roughness, anisotropic, reflect, dir, dir_in, dir_out);

    return f_diffuse + f_sheen + f_metal + f_clearcoat + f_glass;
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_in = dot(frame.n, dir_in);
    //Real n_dot_out = dot(frame.n, dir_out);
    Real n_dot_h = dot(frame.n, half_vector);

    Real w_diffuse   = Real(0);
    Real w_metal     = Real(0);
    Real w_clearcoat = Real(0);
    Real w_glass     = (Real(1) - metallic) * specular_transmission;

    //if (dot(vertex.geometric_normal, dir_in) > 0) {
        w_diffuse   = (Real(1) - metallic) * (Real(1) - specular_transmission);
        w_metal     = (Real(1) - specular_transmission * (Real(1) - metallic));
        w_clearcoat = Real(0.25) * clearcoat;
    //}

    Real w_sum = w_diffuse + w_metal + w_clearcoat + w_glass;

    Real pdf_diffuse = fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
    Real pdf_metal = Real(0);
    //if (n_dot_out > 0 && n_dot_h > 0) {
        // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
        // "Sampling the GGX Distribution of Visible Normals"
        // https://jcgt.org/published/0007/04/01/
        // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
        Real G = smith_masking_gtr2(to_local(frame, dir_in), roughness);
        Real D = GTR2(n_dot_h, roughness);
        // (4 * cos_theta_v) is the Jacobian of the reflectiokn
        pdf_metal = (G * D) / (4 * n_dot_in); 
    //}
    Real alpha_g = (Real(1) - clearcoat_gloss) * Real(0.1) + clearcoat_gloss * Real(0.001);
    Real hl_z = dot(half_vector, frame.n);
    Real D_c = (P2(alpha_g) - Real(1))
             / (c_PI * log(P2(alpha_g)) * (Real(1) + (P2(alpha_g) - Real(1)) * P2(hl_z)));
    // (4 * cos_theta_v) is the Jacobian of the reflectiokn
    Real pdf_clearcoat = (D_c * n_dot_h) / (4 * n_dot_in); 

    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    assert(eta > 0);

    if (!reflect) {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    Real h_dot_in = dot(half_vector, dir_in);
    Real F = fresnel_dielectric(h_dot_in, eta);
    
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_min = Real(0.0001);
    Real aspect = sqrt(Real(1) - Real(0.9) * anisotropic);
    Real alpha_x = max(alpha_min, P2(roughness) / aspect);
    Real alpha_y = max(alpha_min, P2(roughness) * aspect);
    Real hl_x = dot(half_vector, frame.x);
    Real hl_y = dot(half_vector, frame.y);
    hl_z = dot(half_vector, frame.n);
    Real D_denom = c_PI * alpha_x * alpha_y * P2(P2(hl_x / alpha_x) + P2(hl_y / alpha_y) + P2(hl_z));
    D = Real(1) / D_denom;

    Real G_in = smith_masking_gtr2(to_local(frame, dir_in), roughness);
    Real pdf_glass;
    if (reflect) {
        pdf_glass = (F * D * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        Real h_dot_out = dot(half_vector, dir_out);
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        Real dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        pdf_glass = (1 - F) * D * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }

    return (
        w_diffuse   * pdf_diffuse
    +   w_metal     * pdf_metal
    +   w_clearcoat * pdf_clearcoat
    +   w_glass     * pdf_glass
    ) / w_sum;
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real w_diffuse   = Real(0);
    Real w_metal     = Real(0);
    Real w_clearcoat = Real(0);
    Real w_glass     = (Real(1) - metallic) * specular_transmission;

   // if (dot(vertex.geometric_normal, dir_in) > 0) {
        w_diffuse   = (Real(1) - metallic) * (Real(1) - specular_transmission);
        w_metal     = (Real(1) - specular_transmission * (Real(1) - metallic));
        w_clearcoat = Real(0.25) * clearcoat;
    //}

    Real w_accum = Real(0);
    Real w = rnd_param_uv.x * (w_diffuse + w_metal + w_clearcoat + w_glass);

    // Diffuse Lobe
    if (w < (w_accum += w_diffuse)) {
        return BSDFSampleRecord{
            to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
            Real(0) /* eta */, Real(1) /* roughness */};
    }
    // Metal Lobe
    else if (w < (w_accum += w_metal)) {
        // Convert the incoming direction to local coordinates
        Vector3 local_dir_in = to_local(frame, dir_in);
        Real roughness = eval(
            bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
        // Clamp roughness to avoid numerical issues.
        roughness = std::clamp(roughness, Real(0.01), Real(1));
        Real alpha = roughness * roughness;
        Vector3 local_micro_normal =
            sample_visible_normals(local_dir_in, alpha, rnd_param_uv);
        
        // Transform the micro normal to world space
        Vector3 half_vector = to_world(frame, local_micro_normal);
        // Reflect over the world space normal
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        return BSDFSampleRecord{
            reflected, Real(0) /* eta */, roughness /* roughness */
        };
    }
    // Glass Lobe
    else if (w < (w_accum += w_glass)) {
        // If we are going into the surface, then we use normal eta
        // (internal/external), otherwise we use external/internal.
        Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
        
        Real roughness = eval(
            bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
        // Clamp roughness to avoid numerical issues.
        roughness = std::clamp(roughness, Real(0.01), Real(1));
        // Sample a micro normal and transform it to world space -- this is our half-vector.
        Real alpha = roughness * roughness;
        Vector3 local_dir_in = to_local(frame, dir_in);
        Vector3 local_micro_normal =
            sample_visible_normals(local_dir_in, alpha, rnd_param_uv);

        Vector3 half_vector = to_world(frame, local_micro_normal);
        // Flip half-vector if it's below surface
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }

        // Now we need to decide whether to reflect or refract.
        // We do this using the Fresnel term.
        Real h_dot_in = dot(half_vector, dir_in);
        Real F = fresnel_dielectric(h_dot_in, eta);

        if (rnd_param_w <= F) {
            // Reflection
            Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
            // set eta to 0 since we are not transmitting
            return BSDFSampleRecord{reflected, Real(0) /* eta */, roughness};
        } else {
            // Refraction
            // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
            // (note that our eta is eta2 / eta1, and l = -dir_in)
            Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
            if (h_dot_out_sq <= 0) {
                // Total internal reflection
                // This shouldn't really happen, as F will be 1 in this case.
                return {};
            }
            // flip half_vector if needed
            if (h_dot_in < 0) {
                half_vector = -half_vector;
            }
            Real h_dot_out= sqrt(h_dot_out_sq);
            Vector3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
            return BSDFSampleRecord{refracted, eta, roughness};
        }
    }
    // Clearcoat Lobe
    else
        //if (w < (w_accum += w_clearcoat))
    {
        // Convert the incoming direction to local coordinates
        //Vector3 local_dir_in = to_local(frame, dir_in);
        
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
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
