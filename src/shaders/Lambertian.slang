import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_lambertian(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
        dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Sometimes the shading normal can be inconsistent with
    // the geometry normal. We flip the shading frame in that
    // case so that we don't produces "black fringes".
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    return fmax(dot(frame.n, dir_out), 0) *
           eval3(bsdf.reflectance, vertex.uv, vertex.uv_screen_size) / float.getPi();
}

float pdf_lambertian(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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

    // For Lambertian, we importance sample the cosine hemisphere domain.
    return fmax(dot(frame.n, dir_out), 0) / float.getPi();
}

Optional<BSDFSampleRecord> sample_lambertian(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
{
    // For Lambertian, we importance sample the cosine hemisphere domain.
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // Incoming direction is below the surface.
        return none;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    BSDFSampleRecord record = { to_world(frame, sample_cos_hemisphere(rnd_param.xy)) };
    record.eta = 0;
    record.roughness = 1;
    return record;
}