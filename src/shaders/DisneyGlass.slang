import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_disney_glass(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!

    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    float3 base_color = eval3(bsdf.base_color, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);

    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);

    float3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    float h_dot_in = dot(half_vector, dir_in);
    float F = fresnel_dielectric(h_dot_in, eta);
    float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
    float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
    float G_out = smith_masking_gtr2(to_local(frame, dir_out), alpha_x, alpha_y);
    float G = G_in * G_out;

    if (reflect) {
        return base_color * (F * D * G) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        //float eta_factor = dir == TransportDirection::TO_LIGHT ? (1 / (eta * eta)) : 1;
        float eta_factor = (1 / (eta * eta));
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;

        // Burley propose to take the square root of the base color to preserve albedo
        return sqrt(base_color) *
               (eta_factor * (1 - F) * D * G * eta * eta * fabs(h_dot_out * h_dot_in)) /
               (fabs(dot(frame.n, dir_in)) * sqrt_denom * sqrt_denom);
    }
}

float pdf_disney_glass(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                       dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!

    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);

    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);

    float3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    float h_dot_in = dot(half_vector, dir_in);
    float F = fresnel_dielectric(h_dot_in, eta);
    float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
    float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);

    if (reflect) {
        return (F * D * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;
        float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        return (1 - F) * D * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }
}

Optional<BSDFSampleRecord> sample_disney_glass(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
{
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);
    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);

    // Sample a micro normal and transform it to world space -- this is our half-vector.
    float3 local_dir_in = to_local(frame, dir_in);
    float3 local_micro_normal =
        sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd_param.xy);

    float3 half_vector = to_world(frame, local_micro_normal);
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // Now we need to decide whether to reflect or refract.
    // We do this using the Fresnel term.
    float h_dot_in = dot(half_vector, dir_in);
    float F = fresnel_dielectric(h_dot_in, eta);

    if (rnd_param.z <= F) {
        // Reflection
        float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        // set eta to 0 since we are not transmitting
        return BSDFSampleRecord(reflected, 0 /* eta */, roughness);
    } else {
        // Refraction
        // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        // (note that our eta is eta2 / eta1, and l = -dir_in)
        float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
        if (h_dot_out_sq <= 0) {
            // Total internal reflection
            // This shouldn't floatly happen, as F will be 1 in this case.
            return none;
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