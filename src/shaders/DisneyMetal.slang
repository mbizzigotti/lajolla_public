import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_disney_metal(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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
    // Homework 1: implement this!

    // Common variables
    float3 base_color = eval3(bsdf.base_color, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);
    float3 half_vector = normalize(dir_in + dir_out);
    float h_dot_out = dot(half_vector, dir_out);
    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);

    // Disney's metallic BRDF is standard Cook-Torrence
    // microfacet model: F * G * D / (4 * n_dot_in)
    // (We include n_dot_out in our BRDFs)

    // For F, they use a Schlick approximation
    // F0 + (1 - F0) * (1 - h_dot_out)
    // For F0 they blend between base color and RGB(1, 1, 1) using
    // specular and metallic parameters. This does not make much sense
    // in our case since we are only modelling the metallic part --
    // we will use just the base color.
    float3 F = schlick_fresnel(base_color, h_dot_out);

    // For D, they use an anisotropic Trowbridge-Reitz distribution (aka GGX)
    float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);

    // For G, they use the Smith masking term corresponds to GTR2
    float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
    float G_out = smith_masking_gtr2(to_local(frame, dir_out), alpha_x, alpha_y);
    float G = G_in * G_out;

    // And we're done!
    return F * D * G / (4 * dot(dir_in, vertex.shading_frame.n));
}

float pdf_disney_metal(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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
    // Homework 1: implement this!

    // Common variables
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);
    float3 half_vector = normalize(dir_in + dir_out);
    float aspect = sqrt(1 - anisotropic * 0.9);
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);

    // We use visible normal sampling, so the PDF ~ (G_in * D) / (4 * n_dot_in)
    float D = GTR2(to_local(frame, half_vector), alpha_x, alpha_y);
    float G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha_x, alpha_y);
    return D * G_in / (4 * dot(dir_in, vertex.shading_frame.n));
}

Optional<BSDFSampleRecord> sample_disney_metal(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
{
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return none;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!

    // Convert the incoming direction to local coordinates
    float3 local_dir_in = to_local(vertex.shading_frame, dir_in);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float anisotropic = eval1(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size);
    float aspect = sqrt(1 - anisotropic * float(0.9));
    constexpr float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, roughness * roughness / aspect);
    float alpha_y = max(min_alpha, roughness * roughness * aspect);

    float3 local_micro_normal =
        sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd_param.xy);

    // Transform the micro normal to world space
    float3 half_vector = to_world(vertex.shading_frame, local_micro_normal);
    // Reflect over the world space normal
    float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord(reflected, 0 /* eta */, roughness);
}