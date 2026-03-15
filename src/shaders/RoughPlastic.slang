import Shared;
import Intersection;
import Frame;
import Texture;
import Utils;

float3 eval_rough_plastic(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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

    // The half-vector is a crucial component of the microfacet models.
    // Since microfacet assumes that the surface is made of many small mirrors/glasses,
    // The "average" between input and output direction determines the orientation
    // of the mirror our ray hits (since applying reflection of dir_in over half_vector
    // gives us dir_out). Microfacet models build all sorts of quantities based on the
    // half vector. It's also called the "micro normal".
    float3 half_vector = normalize(dir_in + dir_out);
    float n_dot_h = dot(frame.n, half_vector);
    float n_dot_in = dot(frame.n, dir_in);
    float n_dot_out = dot(frame.n, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return 0;
    }

    float3 Kd = eval3(bsdf.diffuse_reflectance, vertex.uv, vertex.uv_screen_size);
    float3 Ks = eval3(bsdf.specular_reflectance, vertex.uv, vertex.uv_screen_size);
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01, 1);

    // We first account for the dielectric layer.

    // Fresnel equation determines how much light goes through,
    // and how much light is reflected for each wavelength.
    // Fresnel equation is determined by the angle between the (micro) normal and
    // both incoming and outgoing directions (dir_out & dir_in).
    // However, since they are related through the Snell-Descartes law,
    // we only need one of them.
    float F_o = fresnel_dielectric(dot(half_vector, dir_out), bsdf.eta); // F_o is the reflection percentage.
    float D = GTR2(n_dot_h, roughness);                                  // "Generalized Trowbridge Reitz", GTR2 is equivalent to GGX.
    float G = smith_masking_gtr2(to_local(frame, dir_in), roughness) *
             smith_masking_gtr2(to_local(frame, dir_out), roughness);

    float3 spec_contrib = Ks * (G * F_o * D) / (4 * n_dot_in * n_dot_out);

    // Next we account for the diffuse layer.
    // In order to reflect from the diffuse layer,
    // the photon needs to bounce through the dielectric layers twice.
    // The transmittance is computed by 1 - fresnel.
    float F_i = fresnel_dielectric(dot(half_vector, dir_in), bsdf.eta);
    // Multiplying with Fresnels leads to an overly dark appearance at the
    // object boundaries. Disney BRDF proposes a fix to this -- we will implement this in problem set 1.
    float3 diffuse_contrib = Kd * (1 - F_o) * (1 - F_i) / float.getPi();

    return (spec_contrib + diffuse_contrib) * n_dot_out;
}

float pdf_rough_plastic(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 dir_out)
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

    float3 half_vector = normalize(dir_in + dir_out);
    float n_dot_in = dot(frame.n, dir_in);
    float n_dot_out = dot(frame.n, dir_out);
    float n_dot_h = dot(frame.n, half_vector);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return 0;
    }

    float3 S = eval3(bsdf.specular_reflectance, vertex.uv, vertex.uv_screen_size);
    float3 R = eval3(bsdf.diffuse_reflectance, vertex.uv, vertex.uv_screen_size);
    float lS = luminance(S), lR = luminance(R);
    if (lS + lR <= 0) {
        return 0;
    }
    float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01, 1);
    // We use the reflectance to determine whether to choose specular sampling lobe or diffuse.
    float spec_prob = lS / (lS + lR);
    float diff_prob = 1 - spec_prob;
    // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
    // "Sampling the GGX Distribution of Visible Normals"
    // https://jcgt.org/published/0007/04/01/
    // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
    float G = smith_masking_gtr2(to_local(frame, dir_in), roughness);
    float D = GTR2(n_dot_h, roughness);
    // (4 * cos_theta_v) is the Jacobian of the reflectiokn
    spec_prob *= (G * D) / (4 * n_dot_in);
    // For the diffuse lobe, we importance sample cos_theta_out
    diff_prob *= n_dot_out / float.getPi();
    return spec_prob + diff_prob;
}

Optional<BSDFSampleRecord> sample_rough_plastic(in Material bsdf, in PathVertex vertex, float3 dir_in, float3 rnd_param)
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

    // We use the reflectance to choose between sampling the dielectric or diffuse layer.
    float3 Ks = eval3(bsdf.specular_reflectance, vertex.uv, vertex.uv_screen_size);
    float3 Kd = eval3(bsdf.diffuse_reflectance, vertex.uv, vertex.uv_screen_size);
    float lS = luminance(Ks), lR = luminance(Kd);
    if (lS + lR <= 0) {
        return {};
    }
    float spec_prob = lS / (lS + lR);
    if (rnd_param.z < spec_prob) {
        // Sample from the specular lobe.

        // Convert the incoming direction to local coordinates
        float3 local_dir_in = to_local(frame, dir_in);
        float roughness = eval1(bsdf.roughness, vertex.uv, vertex.uv_screen_size);
        // Clamp roughness to avoid numerical issues.
        roughness = clamp(roughness, 0.01, 1);
        float alpha = roughness * roughness;
        float3 local_micro_normal =
            sample_visible_normals(local_dir_in, alpha, rnd_param.xy);

        // Transform the micro normal to world space
        float3 half_vector = to_world(frame, local_micro_normal);
        // Reflect over the world space normal
        float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        return BSDFSampleRecord(reflected, 0 /* eta */, roughness);
    } else {
        // Lambertian sampling
        return BSDFSampleRecord(
			to_world(frame, sample_cos_hemisphere(rnd_param.xy)),
            0 /* eta */, 1 /* roughness */);
    }
}