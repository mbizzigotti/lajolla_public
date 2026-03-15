import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_disney_diffuse(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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

    // Disney BSDF's diffuse component contains two parts:
    // 1) a "base" diffuse model based on a modified Fresnel response
    // 2) a "subsurface" diffuse model based on the Lommel-Seeliger law (modified to make things brighter) 
    //    (brought to graphics by Hanrahan & Kruger)

    // Fetch the textures value for later use
    float3 base_color = eval3(bsdf.base_color, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    float subsurface = eval1(bsdf.subsurface, vertex.uv, vertex.uv_screen_size);

    // Common variables
    float n_dot_out = dot(frame.n, dir_out);
    if (n_dot_out <= 0) {
        return 0;
    }
    float n_dot_in = dot(frame.n, dir_in);
    float schlick_n_dot_out = pow(1 - n_dot_out, 5);
    float schlick_n_dot_in  = pow(1 - n_dot_in, 5);
    float3 half_vector = normalize(dir_in + dir_out);
    float h_dot_out = dot(half_vector, dir_out);

    // The base diffuse model
    float Fd90 = 0.5 + 2 * roughness * h_dot_out * h_dot_out;
    float base_diffuse = (1 + (Fd90 - 1) * schlick_n_dot_out) *
                        (1 + (Fd90 - 1) * schlick_n_dot_in);

    // The subsurface model
    // Disney's hack to increase the response at grazing angle
    float Fss90 = h_dot_out * h_dot_out * roughness;
    float Fss = (1 + (Fss90 - 1) * schlick_n_dot_out) *
               (1 + (Fss90 - 1) * schlick_n_dot_in);
    // Lommel-Seeliger law (modified/rescaled)
    float ss = 1.25 * (Fss * (1 / (n_dot_out + n_dot_in) - 0.5) + 0.5);

    return base_color * ((base_diffuse * (1 - subsurface) + ss * subsurface) / float.getPi()) * n_dot_out;
}

float pdf_disney_diffuse(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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
    return fmax(dot(frame.n, dir_out), 0) / float.getPi();
}

Optional<BSDFSampleRecord> sample_disney_diffuse(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
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
    return BSDFSampleRecord(
        to_world(vertex.shading_frame, sample_cos_hemisphere(rnd_param.xy)),
        0 /* eta */, 1 /* roughness */);
}