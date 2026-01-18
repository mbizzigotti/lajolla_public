#include "../microfacet.h"

static Spectrum disney_metal(
    Frame    frame,
    Spectrum F_m,
    Real     roughness,
    Real     anisotropic,
    Vector3  half_vector,
    Vector3  dir_in,
    Vector3  dir_out
)
{   
    Real alpha_min = Real(0.0001);
    Real aspect = sqrt(Real(1) - Real(0.9) * anisotropic);
    Real alpha_x = max(alpha_min, P2(roughness) / aspect);
    Real alpha_y = max(alpha_min, P2(roughness) * aspect);
    Real hl_x = dot(half_vector, frame.x);
    Real hl_y = dot(half_vector, frame.y);
    Real hl_z = dot(half_vector, frame.n);
    Real D_m = c_PI * alpha_x * alpha_y * P2(P2(hl_x / alpha_x) + P2(hl_y / alpha_y) + P2(hl_z));
    D_m = Real(1) / D_m;

    auto G = [&](Vector3 w) {
        Real wl_x = dot(w, frame.x);
        Real wl_y = dot(w, frame.y);
        Real wl_z = dot(w, frame.n);
        Real x = (P2(wl_x * alpha_x) + P2(wl_y * alpha_y)) / P2(wl_z);
        Real V = Real(0.5) * (sqrt(Real(1) + x) - Real(1));
        return Real(1) / (Real(1) + V);
    };
    Real G_m = G(dir_in) * G(dir_out);
    
    return F_m * D_m * G_m / (Real(4) * abs(dot(frame.n, dir_in)));
}

static Real disney_metal_pdf(
    PathVertex const& vertex,
    Vector3 dir_in,
    Vector3 dir_out,
    Real roughness,
    Real anisotropic
)
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        //assert(false && "METAL");
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    Real n_dot_h = dot(frame.n, half_vector);
    //if (n_dot_out <= 0 || n_dot_h <= 0) {
    //    return 0;
    //}
    
    // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
    // "Sampling the GGX Distribution of Visible Normals"
    // https://jcgt.org/published/0007/04/01/
    // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
    Real alpha_min = Real(0.0001);
    Real aspect = sqrt(Real(1) - Real(0.9) * anisotropic);
    Real alpha_x = max(alpha_min, P2(roughness) / aspect);
    Real alpha_y = max(alpha_min, P2(roughness) * aspect);
    Real hl_x = dot(half_vector, frame.x);
    Real hl_y = dot(half_vector, frame.y);
    Real hl_z = dot(half_vector, frame.n);
    Real D_m = c_PI * alpha_x * alpha_y * P2(P2(hl_x / alpha_x) + P2(hl_y / alpha_y) + P2(hl_z));
    D_m = Real(1) / D_m;

    auto G = [&](Vector3 w) {
        Real wl_x = dot(w, frame.x);
        Real wl_y = dot(w, frame.y);
        Real wl_z = dot(w, frame.n);
        Real x = (P2(wl_x * alpha_x) + P2(wl_y * alpha_y)) / P2(wl_z);
        Real V = Real(0.5) * (sqrt(Real(1) + x) - Real(1));
        return Real(1) / (Real(1) + V);
    };
    Real G_m = G(dir_in) * G(dir_out);
    // (4 * cos_theta_v) is the Jacobian of the reflectiokn
    return (G_m * D_m) / (4 * n_dot_in); 
}

static BSDFSampleRecord disney_metal_sample(
    PathVertex const& vertex,
    Vector3 dir_in,
    Vector2 rnd_param_uv,
    Real roughness,
    Real anisotropic
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

    // Convert the incoming direction to local coordinates
    Vector3 local_dir_in = to_local(frame, dir_in);
    // Clamp roughness to avoid numerical issues.
    //roughness = std::clamp(roughness, Real(0.001), Real(1));
    Real aspect = sqrt(Real(1) - Real(0.9) * anisotropic);
    Vector2 alpha = {
        max(Real(0.0001), roughness * roughness / aspect),
        max(Real(0.0001), roughness * roughness / aspect),
    };
    Vector3 local_micro_normal =
        sample_visible_normals_anisotropic(local_dir_in, alpha, rnd_param_uv);
    
    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, roughness /* roughness */
    };
}

Spectrum eval_op::operator()(const DisneyMetal &bsdf) const {
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
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_vector = normalize(dir_in + dir_out);
    Spectrum F_m = base_color + (Real(1) - base_color) * P5(Real(1) - abs(dot(half_vector, dir_out)));
    return disney_metal(frame, F_m, roughness, anisotropic, half_vector, dir_in, dir_out);
}

Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    return disney_metal_pdf(vertex, dir_in, dir_out, roughness, anisotropic);  
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    return disney_metal_sample(vertex, dir_in, rnd_param_uv, roughness, anisotropic);  
}

TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const {
    return bsdf.base_color;
}
