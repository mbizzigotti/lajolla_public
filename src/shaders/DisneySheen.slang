import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_disney_sheen(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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

    float3 base_color = eval3(bsdf.base_color, vertex.uv, vertex.uv_screen_size);
    float sheen_tint = eval1(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size);

    float3 half_vector = normalize(dir_in + dir_out);
    float h_dot_out = dot(half_vector, dir_out);
    float3 Ctint = luminance(base_color) > 0 ? base_color / luminance(base_color) : 1;
    float3 Csheen = (1 - sheen_tint) + sheen_tint * Ctint;
    return Csheen * pow(1 - h_dot_out, 5) * fmax(dot(frame.n, dir_out), 0);
}

float pdf_disney_sheen(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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

    // We sample the cosine hemisphere
    return fmax(dot(frame.n, dir_out), 0) / float.getPi();
}

Optional<BSDFSampleRecord> sample_disney_sheen(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
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

    // We sample the cosine hemisphere
    return BSDFSampleRecord(
        to_world(vertex.shading_frame, sample_cos_hemisphere(rnd_param.xy)),
        0 /* eta */, 1 /* roughness */);
}