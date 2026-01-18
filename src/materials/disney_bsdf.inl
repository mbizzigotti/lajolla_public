#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {
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

    // Clamp roughness to avoid numerical issues.
    //roughness = std::clamp(roughness, Real(0.001), Real(1));

    if (dot(vertex.geometric_normal, dir_in) > 0
    &&  dot(vertex.geometric_normal, dir_out) > 0) {
        // Flip the shading frame if it is inconsistent with the geometry normal
        Frame frame = vertex.shading_frame;
        if (dot(frame.n, dir_in) < 0) {
            frame = -frame;
        }

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
        Spectrum C_0 = specular * R_0(eta) * (Real(1) - metallic) * K_s + metallic * base_color;
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

    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    Spectrum f_glass = (Real(1) - metallic) * specular_transmission
                     * disney_glass(frame, base_color, eta, roughness, anisotropic, reflect, dir, dir_in, dir_out);

    return f_diffuse + f_sheen + f_metal + f_clearcoat + f_glass;
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    // Clamp roughness to avoid numerical issues.
    //roughness = std::clamp(roughness, Real(0.001), Real(1));

    Real w_diffuse   = Real(0);
    Real w_metal     = Real(0);
    Real w_clearcoat = Real(0);
    Real w_glass     = Real(1);

    Real pdf_diffuse   = Real(0);
    Real pdf_metal     = Real(0);
    Real pdf_clearcoat = Real(0);

    if (dot(vertex.geometric_normal, dir_in) > 0
    &&  dot(vertex.geometric_normal, dir_out) > 0) {
        w_diffuse   = (Real(1) - metallic) * (Real(1) - specular_transmission);
        w_metal     = (Real(1) - specular_transmission * (Real(1) - metallic));
        w_clearcoat = Real(0.25) * clearcoat;
        w_glass     = (Real(1) - metallic) * specular_transmission;

        pdf_diffuse   = disney_diffuse_pdf(vertex, dir_in, dir_out);
        pdf_metal     = disney_metal_pdf(vertex, dir_in, dir_out, roughness, anisotropic);
        pdf_clearcoat = disney_clearcoat_pdf(vertex, dir_in, dir_out, clearcoat_gloss);
    }
    Real pdf_glass = disney_glass_pdf(vertex, dir_in, dir_out, bsdf.eta, roughness, anisotropic);

    return (
        w_diffuse   * pdf_diffuse
    +   w_metal     * pdf_metal
    +   w_clearcoat * pdf_clearcoat
    +   w_glass     * pdf_glass
    ) / (w_diffuse + w_metal + w_clearcoat + w_glass);
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real w_diffuse   = Real(0);
    Real w_metal     = Real(0);
    Real w_clearcoat = Real(0);
    Real w_glass     = Real(1);

    if (dot(vertex.geometric_normal, dir_in) > 0) {
        w_diffuse   = (Real(1) - metallic) * (Real(1) - specular_transmission);
        w_metal     = (Real(1) - specular_transmission * (Real(1) - metallic));
        w_clearcoat = Real(0.25) * clearcoat;
        w_glass     = (Real(1) - metallic) * specular_transmission;
    }

    Real w_accum = Real(0);
    Real w = rnd_param_w * (w_diffuse + w_metal + w_clearcoat + w_glass);

    // Glass Lobe
    if (w < (w_accum += w_glass)) {
        return disney_glass_sample(vertex, dir_in, rnd_param_uv, rnd_param_w / w_glass, bsdf.eta, roughness, anisotropic);
    }
    // Metal Lobe
    if (w < (w_accum += w_metal)) {
        return disney_metal_sample(vertex, dir_in, rnd_param_uv, roughness, anisotropic);
    }
    // Diffuse Lobe
    if (w < (w_accum += w_diffuse)) {
        return disney_diffuse_sample(vertex, dir_in, rnd_param_uv);
    }
    // Clearcoat Lobe
    if (w < (w_accum += w_clearcoat)) {
        return disney_clearcoat_sample(vertex, dir_in, rnd_param_uv, clearcoat_gloss);
    }

    assert(false && "unreachable");
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
