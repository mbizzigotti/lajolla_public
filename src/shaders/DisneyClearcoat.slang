import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_disney_clearcoat(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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
    float clearcoat_gloss = eval1(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size);
    float alpha = (1 - clearcoat_gloss) * float(0.1) + clearcoat_gloss * float(0.001);

    float3 half_vector = normalize(dir_in + dir_out);
    float n_dot_h = dot(frame.n, half_vector);
    float n_dot_out = dot(frame.n, dir_out);
    float h_dot_out = dot(half_vector, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0 || h_dot_out <= 0) {
        return 0;
    }

    // They use a hardcoded IOR 1.5 -> F0 = 0.04
    float F = schlick_fresnel(0.04, h_dot_out);
    // Generalized Trowbridge-Reitz distribution
    float D = GTR1(n_dot_h, alpha);
    // SmithG with fixed alpha
    float G_in = smith_masking_gtr1(to_local(frame, dir_in));
    float G_out = smith_masking_gtr1(to_local(frame, dir_out));
    float G = G_in * G_out;

    return F * D * G / (4 * dot(dir_in, vertex.shading_frame.n));
}

float pdf_disney_clearcoat(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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

    float clearcoat_gloss = eval1(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size);
    float alpha = (1 - clearcoat_gloss) * float(0.1) + clearcoat_gloss * float(0.001);

    float3 half_vector = normalize(dir_in + dir_out);
    float n_dot_h = dot(frame.n, half_vector);
    float n_dot_out = dot(frame.n, dir_out);
    float h_dot_out = dot(half_vector, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0 || h_dot_out <= 0) {
        return 0;
    }

    // We only importance sample D
    float D = GTR1(n_dot_h, alpha);

    return D * n_dot_h / (4 * h_dot_out);
}

Optional<BSDFSampleRecord> sample_disney_clearcoat(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
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
    // Homework 1: implement this!

    // Since the clearcoat BRDF does not have a clear geometry meaning,
    // it is hard to sample the visible normals. We will just importance sample D
    float clearcoat_gloss = eval1(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size);
    float alpha = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;

    // Appendix B.2 Burley's note
    float alpha2 = alpha * alpha;
    // Equation 5
    float cos_h_elevation =
        sqrt(max(float(0), (1 - pow(alpha2, 1 - rnd_param[0])) / (1 - alpha2)));
    float sin_h_elevation = sqrt(max(1 - cos_h_elevation * cos_h_elevation, 0));
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
    return BSDFSampleRecord(reflected, 0 /* eta */, sqrt(alpha) /* roughness */);
}