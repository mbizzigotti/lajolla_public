import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_rough_dielectric(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    float3 Ks = eval3(bsdf.specular_reflectance, vertex.uv, vertex.uv_screen_size);
    float3 Kt = eval3(bsdf.specular_transmittance, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);

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

    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.02, 1);

    // Compute F / D / G
    // Note that we use the incoming direction
    // for evaluating the Fresnel reflection amount.
    // We can also use outgoing direction -- then we would need to
    // use 1/bsdf.eta and we will get the same result.
    // However, using the incoming direction allows
    // us to use F to decide whether to reflect or refract during sampling.
    float h_dot_in = dot(half_vector, dir_in);
    float F = fresnel_dielectric(h_dot_in, eta);
    float D = GTR2(dot(frame.n, half_vector), roughness);
    float G = smith_masking_gtr2(to_local(frame, dir_in), roughness) *
             smith_masking_gtr2(to_local(frame, dir_out), roughness);
    if (reflect) {
        return Ks * (F * D * G) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        // Snell-Descartes law predicts that the light will contract/expand
        // due to the different index of refraction. So the normal BSDF needs
        // to scale with 1/eta^2. However, the "adjoint" of the BSDF does not have
        // the eta term. This is due to the non-reciprocal nature of the index of refraction:
        // f(wi -> wo) / eta_o^2 = f(wo -> wi) / eta_i^2
        // thus f(wi -> wo) = f(wo -> wi) (eta_o / eta_i)^2
        // The adjoint of a BSDF is defined as swapping the parameter, and
        // this cancels out the eta term.
        // See Chapter 5 of Eric Veach's thesis "Robust Monte Carlo Methods for Light Transport Simulation"
        // for more details.
        //float eta_factor = dir == TransportDirection::TO_LIGHT ? (1 / (eta * eta)) : 1;
        float eta_factor = (1 / (eta * eta));
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;
        // Very complicated BSDF. See Walter et al.'s paper for more details.
        // "Microfacet Models for Refraction through Rough Surfaces"
        return Kt * (eta_factor * (1 - F) * D * G * eta * eta * fabs(h_dot_out * h_dot_in)) /
               (fabs(dot(frame.n, dir_in)) * sqrt_denom * sqrt_denom);
    }
}

float pdf_rough_dielectric(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
{
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    // assert(eta > 0);

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

    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.02, 1);

    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    float h_dot_in = dot(half_vector, dir_in);
    float F = fresnel_dielectric(h_dot_in, eta);
    float D = GTR2(dot(half_vector, frame.n), roughness);
    float G_in = smith_masking_gtr2(to_local(frame, dir_in), roughness);
    if (reflect) {
        return (F * D * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;
        float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        return (1 - F) * D * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }
}

Optional<BSDFSampleRecord> sample_rough_dielectric(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
{
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    float eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.02, 1);
    // Sample a micro normal and transform it to world space -- this is our half-vector.
    float alpha = roughness * roughness;
    float3 local_dir_in = to_local(frame, dir_in);
    float3 local_micro_normal =
        sample_visible_normals(local_dir_in, alpha, rnd_param.xy);

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
            // This shouldn't really happen, as F will be 1 in this case.
            return {};
        }
        // flip half_vector if needed
        if (h_dot_in < 0) {
            half_vector = -half_vector;
        }
        float h_dot_out= sqrt(h_dot_out_sq);
        float3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
        return BSDFSampleRecord(refracted, eta, roughness);
    }
}