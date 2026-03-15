import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_disney_bsdf(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!

    // Fetch the texture values for later use
    float3 base_color = eval3(bsdf.base_color, vertex.uv, vertex.uv_screen_size);
    float specular_transmission = eval1(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size);
    float metallic = eval1(bsdf.metallic, vertex.uv, vertex.uv_screen_size);
    float subsurface = eval1(bsdf.subsurface, vertex.uv, vertex.uv_screen_size);
    float specular = eval1(bsdf.specular, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float specular_tint = eval1(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);
    float sheen = eval1(bsdf.sheen, vertex.uv, vertex.uv_screen_size);
    float sheen_tint = eval1(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size);
    float clearcoat = 0.25 * eval1(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size);
    float clearcoat_gloss = eval1(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size);

    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);
    float alpha_c = (1 - clearcoat_gloss) * float(0.1) + clearcoat_gloss * float(0.001);

    if (reflect) {
        // Diffuse component
        float n_dot_in = dot(frame.n, dir_in);
        float n_dot_out = dot(frame.n, dir_out);
        float3 half_vector = normalize(dir_in + dir_out);
        // Flip half-vector if it's below surface
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }
        float n_dot_h = dot(half_vector, vertex.shading_frame.n);
        float h_dot_in = dot(half_vector, dir_in);
        float h_dot_out = dot(half_vector, dir_out);

        float3 diffuse_contrib = 0;
        float3 metallic_contrib = 0;
        float clearcoat_contrib = 0;
        float3 sheen_contrib = 0;

        // For diffuse, metallic, sheen, and clearcoat, the light bounces
        // only at the top of the surface.
        if (dot(vertex.geometric_normal, dir_in) >= 0 &&
            dot(vertex.geometric_normal, dir_out) >= 0 &&
            n_dot_out > 0) {
            // Diffuse component

            // The base diffuse model
            float Fd90 = float(0.5) + 2 * roughness * h_dot_out * h_dot_out;
            float schlick_n_dot_out = pow(1 - n_dot_out, float(5));
            float schlick_n_dot_in = pow(1 - n_dot_in, float(5));
            float schlick_h_dot_out = pow(1 - h_dot_out, float(5));
            float base_diffuse = (1 + (Fd90 - 1) * schlick_n_dot_out) *
                                (1 + (Fd90 - 1) * schlick_n_dot_in);

            // The subsurface model
            // Disney's hack to increase the response at grazing angle
            float Fss90 = h_dot_out * h_dot_out * roughness;
            float Fss = (1 + (Fss90 - 1) * schlick_n_dot_out) *
                       (1 + (Fss90 - 1) * schlick_n_dot_in);
            // Lommel-Seeliger law (modified/rescaled)
            float ss = float(1.25) * (Fss * (1 / (n_dot_out + n_dot_in) - float(0.5)) + float(0.5));

            diffuse_contrib =
                (1 - specular_transmission) * (1 - metallic) * base_color *
                ((base_diffuse * (1 - subsurface) + ss * subsurface) / float.getPi()) * n_dot_out;

            // Sheen component
            float3 Ctint =
                luminance(base_color) > 0 ? base_color / luminance(base_color) : 1;
            float3 Csheen = (1 - sheen_tint) + sheen_tint * Ctint;
            sheen_contrib = (1 - metallic) * sheen * Csheen * schlick_h_dot_out * n_dot_out;

            // Metallic component
            if (n_dot_in > 0 && h_dot_out > 0 && n_dot_h > 0) {
                float eta = bsdf.eta; // we're always going inside
                float spec_f0 = (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
                float3 spec_color = ((1 - specular_tint) + specular_tint * Ctint);
                float3 Cspec0 =
                    specular * spec_f0 * (1 - metallic) * spec_color + metallic * base_color;
                float spec_weight = (1 - specular_transmission * (1 - metallic));

                float3 F = schlick_fresnel(Cspec0, h_dot_out);
                float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
                float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
                float G_out = smith_masking_gtr2(to_local(frame, dir_out), alpha_x, alpha_y);
                float G = G_in * G_out;
                metallic_contrib = spec_weight * F * D * G / (4 * n_dot_in);
            }

            // Clearcoat component
            if (n_dot_in > 0 && n_dot_h > 0) {
                float Fc = schlick_fresnel(float(0.04), h_dot_out);
                // Generalized Trowbridge-Reitz distribution
                float Dc = GTR1(n_dot_h, alpha_c);
                // SmithG with fixed alpha
                float Gc_in = smith_masking_gtr1(to_local(frame, dir_in));
                float Gc_out = smith_masking_gtr1(to_local(frame, dir_out));
                float Gc = Gc_in * Gc_out;

                clearcoat_contrib = clearcoat * Fc * Dc * Gc / (4 * n_dot_in);
            }
        }

        // For glass, lights bounce at both sides of the surface.

        // Glass component
        float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
        float Fg = fresnel_dielectric(h_dot_in, eta);
        float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
        float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
        float G_out = smith_masking_gtr2(to_local(frame, dir_out), alpha_x, alpha_y);
        float G = G_in * G_out;
        float3 glass_contrib =
            base_color * ((1 - metallic) * specular_transmission * (Fg * D * G) / (4 * fabs(n_dot_in)));

        return diffuse_contrib + sheen_contrib + metallic_contrib + glass_contrib + clearcoat_contrib;
    } else {
        // Only the glass component for refraction
        float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
        float3 half_vector = normalize(dir_in + dir_out * eta);
        // Flip half-vector if it's below surface
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }

//        float eta_factor = dir == TransportDirection::TO_LIGHT ? (1 / (eta * eta)) : 1;
        float eta_factor = (1 / (eta * eta));
        float h_dot_in = dot(half_vector, dir_in);
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;

        float Fg = fresnel_dielectric(h_dot_in, eta);
        float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
        float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
        float G_out = smith_masking_gtr2(to_local(frame, dir_out), alpha_x, alpha_y);
        float G = G_in * G_out;

        // Burley propose to take the square root of the base color to preserve albedo
        return sqrt(base_color) * ((1 - metallic) * specular_transmission *
                                   (eta_factor * (1 - Fg) * D * G * eta * eta * fabs(h_dot_out * h_dot_in)) /
                                   (fabs(dot(frame.n, dir_in)) * sqrt_denom * sqrt_denom));
    }
}

float pdf_disney_bsdf(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    // Homework 1: implement this!

    // Fetch the texture values for later use
    float specular_transmission = eval1(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size);
    float metallic = eval1(bsdf.metallic, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);
    float clearcoat = 0.25 * eval1(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size);
    float clearcoat_gloss = eval1(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size);

    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);
    float alpha_c = (1 - clearcoat_gloss) * float(0.1) + clearcoat_gloss * float(0.001);

    float diffuse_weight = (1 - metallic) * (1 - specular_transmission);
    float metallic_weight = (1 - specular_transmission * (1 - metallic));
    float glass_weight = (1 - metallic) * specular_transmission;
    float clearcoat_weight = clearcoat;
    float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
    float diffuse_prob = diffuse_weight / total_weight;
    float metallic_prob = metallic_weight / total_weight;
    float glass_prob = glass_weight / total_weight;
    float clearcoat_prob = clearcoat_weight / total_weight;

    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // Our incoming ray is coming from inside,
        // so the probability of sampling the glass lobe is 1 if glass_prob is not 0.
        diffuse_prob = 0;
        metallic_prob = 0;
        clearcoat_prob = 0;
        if (glass_prob > 0) {
            glass_prob = 1;
        }
    }

    if (reflect) {
        // For metallic: visible normal sampling -> D * G_in
        float3 half_vector = normalize(dir_in + dir_out);
        // Flip half-vector if it's below surface
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }
        float n_dot_in = dot(frame.n, dir_in);
        float n_dot_h = dot(half_vector, vertex.shading_frame.n);
        float h_dot_in = dot(half_vector, dir_in);
        float h_dot_out = dot(half_vector, dir_out);

        // For diffuse, metallic, and clearcoat, the light bounces
        // only at the top of the surface.
        if (dot(vertex.geometric_normal, dir_in) >= 0 &&
            dot(vertex.geometric_normal, dir_out) >= 0) {
            diffuse_prob *= fmax(dot(frame.n, dir_out), float(0)) / float.getPi();

            if (n_dot_in > 0) {
                float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
                float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
                metallic_prob *= (D * G_in / (4 * n_dot_in));
            } else {
                metallic_prob = 0;
            }

            // For clearcoat: D importance sampling
            if (n_dot_h > 0 && h_dot_out > 0) {
                float Dc = GTR1(n_dot_h, alpha_c);
                clearcoat_prob *= (Dc * n_dot_h / (4 * h_dot_out));
            } else {
                clearcoat_prob = 0;
            }
        }

        // For glass: F * visible normal
        float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
        float Fg = fresnel_dielectric(h_dot_in, eta);
        float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
        float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
        glass_prob *= (Fg * D * G_in / (4 * fabs(n_dot_in)));
    } else {
        // Only glass component for refraction
        float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
        float3 half_vector = normalize(dir_in + dir_out * eta);
        // Flip half-vector if it's below surface
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }
        float h_dot_in = dot(half_vector, dir_in);
        float h_dot_out = dot(half_vector, dir_out);
        float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
        float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
        float Fg = fresnel_dielectric(h_dot_in, eta);
        float sqrt_denom = h_dot_in + eta * h_dot_out;
        float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        glass_prob *= (1 - Fg) * D * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }

    return diffuse_prob + metallic_prob + glass_prob + clearcoat_prob;
}

Optional<BSDFSampleRecord> sample_disney_bsdf(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
{
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    // Homework 1: implement this!

    // Fetch the texture values for later use
    float specular_transmission = eval1(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size);
    float metallic = eval1(bsdf.metallic, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);
    float clearcoat = 0.25 * eval1(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size);
    float clearcoat_gloss = eval1(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size);

    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);
    float alpha_c = (1 - clearcoat_gloss) * float(0.1) + clearcoat_gloss * float(0.001);

    float diffuse_weight = (1 - metallic) * (1 - specular_transmission);
    float metallic_weight = (1 - specular_transmission * (1 - metallic));
    float glass_weight = (1 - metallic) * specular_transmission;
    float clearcoat_weight = clearcoat;

    // Two cases: 1) if we are coming from "outside" the surface,
    // sample all lobes
    if (dot(vertex.geometric_normal, dir_in) >= 0) {
        float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
        float diffuse_prob = diffuse_weight / total_weight;
        float metallic_prob = metallic_weight / total_weight;
        float glass_prob = glass_weight / total_weight;
        // float clearcoat_prob = clearcoat_weight / total_weight;
        if (rnd_param.z <= diffuse_prob) {
            return BSDFSampleRecord(
                to_world(vertex.shading_frame, sample_cos_hemisphere(rnd_param.xy)),
                0 /* eta */, 1 /* roughness */);
        } else if (rnd_param.z <= (diffuse_prob + metallic_prob)) { // metallic
            // Visible normal sampling

            // Convert the incoming direction to local coordinates
            float3 local_dir_in = to_local(vertex.shading_frame, dir_in);
            float3 local_micro_normal =
                sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd_param.xy);

            // Transform the micro normal to world space
            float3 half_vector = to_world(vertex.shading_frame, local_micro_normal);
            // Reflect over the world space normal
            float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
            return BSDFSampleRecord(reflected, 0 /* eta */, roughness);
        } else if (rnd_param.z <= (diffuse_prob + metallic_prob + glass_prob)) { // glass
            if (glass_prob <= 0) {
                // Just to be safe numerically.
                return none;
            }
            // Visible normal sampling

            // Convert the incoming direction to local coordinates
            float3 local_dir_in = to_local(vertex.shading_frame, dir_in);
            float3 local_micro_normal =
                sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd_param.xy);

            // Transform the micro normal to world space
            float3 half_vector = to_world(vertex.shading_frame, local_micro_normal);
            // Flip half-vector if it's below surface
            if (dot(half_vector, frame.n) < 0) {
                half_vector = -half_vector;
            }

            // Now we need to decide whether to reflect or refract.
            // We do this using the Fresnel term.
            float h_dot_in = dot(half_vector, dir_in);
            float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
            float F = fresnel_dielectric(h_dot_in, eta);
            // rescale rnd_param.z from
            // (diffuse_prob + metallic_prob, diffuse_prob + metallic_prob + glass_prob]
            // to
            // (0, 1]
            float u = (rnd_param.z - (diffuse_prob + metallic_prob)) / glass_prob;
            if (u <= F) {
                // Reflect over the world space normal
                float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
                return BSDFSampleRecord(reflected, 0 /* eta */, roughness);
            } else {
                // Refraction
                float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
                if (h_dot_out_sq <= 0) {
                    return {};
                }
                // flip half_vector if needed
                if (h_dot_in < 0) {
                    half_vector = -half_vector;
                }
                float h_dot_out = sqrt(h_dot_out_sq);
                float3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
                return BSDFSampleRecord(refracted, eta, roughness);
            }
        } else { // clearcoat
            // Only importance sampling D

            // Appendix B.2 Burley's note
            float alpha2 = alpha_c * alpha_c;
            // Equation 5
            float cos_h_elevation =
                sqrt(max(float(0), (1 - pow(alpha2, 1 - rnd_param[0])) / (1 - alpha2)));
            float sin_h_elevation = sqrt(max(1 - cos_h_elevation * cos_h_elevation, float(0)));
            float h_azimuth = 2 * float.getPi() * rnd_param[1];
            float3 local_micro_normal = {
                sin_h_elevation * cos(h_azimuth),
                sin_h_elevation * sin(h_azimuth),
                cos_h_elevation
            };
            // Transform the micro normal to world space
            float3 half_vector = to_world(frame, local_micro_normal);

            // Reflect over the world space normal
            float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
            return BSDFSampleRecord(reflected, 0 /* eta */, sqrt(alpha_c) /* roughness */);
        }
    } else {
        // 2) otherwise, only consider the glass lobes.

        // Convert the incoming direction to local coordinates
        float3 local_dir_in = to_local(vertex.shading_frame, dir_in);
        float3 local_micro_normal =
            sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd_param.xy);

        // Transform the micro normal to world space
        float3 half_vector = to_world(vertex.shading_frame, local_micro_normal);
        // Flip half-vector if it's below surface
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }

        // Now we need to decide whether to reflect or refract.
        // We do this using the Fresnel term.
        float h_dot_in = dot(half_vector, dir_in);
        float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
        float F = fresnel_dielectric(h_dot_in, eta);
        float u = rnd_param.z;
        if (u <= F) {
            // Reflect over the world space normal
            float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
            return BSDFSampleRecord(reflected, 0 /* eta */, roughness);
        } else {
            // Refraction
            float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
            if (h_dot_out_sq <= 0) {
                return {};
            }
            // flip half_vector if needed
            if (h_dot_in < 0) {
                half_vector = -half_vector;
            }
            float h_dot_out = sqrt(h_dot_out_sq);
            float3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
            return BSDFSampleRecord(refracted, eta, roughness);
        }
    }
}