
[ForceInline]
float distance_squared(float3 a, float3 b)
{
    float3 diff = b - a;
    return dot(diff, diff);
}

[ForceInline]
float luminance(float3 s)
{
    return dot(s, float3(0.212671, 0.715160, 0.072169));
}

[ForceInline]
func max_comp(float3 v) -> float
{
    return max(v.x, max(v.y, v.z));
}

[ForceInline]
func safe_sqrt(float x) -> float
{
    return sqrt(x);
}

[ForceInline]
float3 xform_point(float4x4 xform, float3 pt)
{
    float4 tpt = mul(transpose(xform), float4(pt, 1));
    float inv_w = 1 / tpt[3];
    return float3(tpt[0] * inv_w, tpt[1] * inv_w, tpt[2] * inv_w);
}

[ForceInline]
float3 xform_vector(float4x4 xform, float3 vec)
{
    return mul(transpose(xform), float4(vec, 0)).xyz;
}

[ForceInline]
float3 sample_cos_hemisphere(float2 rnd_param)
{
	float phi = 2 * float.getPi() * rnd_param[0];
    float tmp = sqrt(saturate(1 - rnd_param[1]));
    return {
        cos(phi) * tmp, sin(phi) * tmp,
        sqrt(saturate(rnd_param[1]))
    };
}

[ForceInline]
float3 sample_half_vector(float alpha, float2 rnd_param)
{
    float a2 = alpha * alpha;
    float cos_h_elevation = sqrt(
        (float(1) - pow(a2, float(1) - rnd_param.x))
      / (float(1) - a2));
    float h_azimuth = 2 * float.getPi() * rnd_param.y;
    float sin_h_azimuth = sin(h_azimuth);
    float cos_h_azimuth = cos(h_azimuth);
    float sin_h_elevation = sqrt(float(1) - cos_h_elevation * cos_h_elevation);
    return float3(
        sin_h_elevation * cos_h_azimuth,
        sin_h_elevation * sin_h_azimuth,
        cos_h_elevation
    );
}

/// Fresnel equation of a dielectric interface.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// n_dot_i: abs(cos(incident angle))
/// n_dot_t: abs(cos(transmission angle))
/// eta: eta_transmission / eta_incident
[ForceInline]
float fresnel_dielectric(float n_dot_i, float n_dot_t, float eta) {
    // assert(n_dot_i >= 0 && n_dot_t >= 0 && eta > 0);
    float rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    float rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    float F = (rs * rs + rp * rp) / 2;
    return F;
}

/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// This is a specialized version for the code above, only using the incident angle.
/// The transmission angle is derived from
/// n_dot_i: cos(incident angle) (can be negative)
/// eta: eta_transmission / eta_incident
[ForceInline]
float fresnel_dielectric(float n_dot_i, float eta) {
    // assert(eta > 0);
    float n_dot_t_sq = 1 - ((1 - n_dot_i * n_dot_i) / (eta * eta));
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1;
    }
    float n_dot_t = safe_sqrt(n_dot_t_sq);
    return fresnel_dielectric(fabs(n_dot_i), n_dot_t, eta);
}

/// Schlick's Fresnel equation approximation
/// from "An Inexpensive BRDF Model for Physically-based Rendering", Schlick
/// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297&rep=rep1&type=pdf
/// See "Memo on Fresnel equations" from Sebastien Lagarde
/// for a really nice introduction.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
[ForceInline]
func schlick_fresnel(float F0, float cos_theta) -> float {
    return F0 + (1.0 - F0) * pow(max(1 - cos_theta, 0), 5);
}
[ForceInline]
func schlick_fresnel(float3 F0, float cos_theta) -> float3 {
    return F0 + (1.0 - F0) * pow(max(1 - cos_theta, 0), 5);
}
[ForceInline]
float3 schlick_fresnel(float3 F0, float cos_theta, float eta) {
    float h_dot_out_sq = 1 - (1 / (eta * eta)) * (1 - cos_theta * cos_theta);
    float3 F = 1;
    if (h_dot_out_sq > 0) {
        F = schlick_fresnel(F0, eta > 1 ? cos_theta : sqrt(h_dot_out_sq));
    }
    return F;
}

[ForceInline]
float GTR1(float n_dot_h, float alpha) {
    float a2 = alpha * alpha;
    float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return (a2 - 1) / (float.getPi() * log(a2) * t);
}

[ForceInline]
float GTR2(float n_dot_h, float roughness) {
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (float.getPi() * t * t);
}

[ForceInline]
float GTR2(float3 h_local, float alpha_x, float alpha_y) {
    float3 h_local_scaled = {h_local.x / alpha_x, h_local.y / alpha_y, h_local.z};
    float h_local_scaled_len_sq = dot(h_local_scaled, h_local_scaled);
    return 1 / (float.getPi() * alpha_x * alpha_y * h_local_scaled_len_sq * h_local_scaled_len_sq);
}

[ForceInline]
float GGX(float n_dot_h, float roughness) {
    return GTR2(n_dot_h, roughness);
}

/// The masking term models the occlusion between the small mirrors of the microfacet models.
/// See Eric Heitz's paper "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
/// for a great explanation.
/// https://jcgt.org/published/0003/02/03/paper.pdf
/// The derivation is based on Smith's paper "Geometrical shadowing of a random rough surface".
/// Note that different microfacet distributions have different masking terms.
[ForceInline]
float smith_masking_gtr2(float3 v_local, float roughness) {
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float3 v2 = v_local * v_local;
    float Lambda = (-1 + safe_sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

[ForceInline]
float smith_masking_gtr2(float3 v_local, float alpha_x, float alpha_y) {
    float ax2 = alpha_x * alpha_x;
    float ay2 = alpha_y * alpha_y;
    float3 v2 = v_local * v_local;
    float Lambda = (-1 + sqrt(1 + (v2.x * ax2 + v2.y * ay2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

[ForceInline]
float smith_masking_gtr1(float3 v_local) {
    return smith_masking_gtr2(v_local, 0.25, 0.25);
}

int upper_bound(ByteAddressBuffer arr, int offset, int count, float value)
{
    int first = offset;
    int count = count;
    while (0 < count) { // divide and conquer, find half that contains answer
		int count2 = count / 2;
        int mid = first + count2;
        float mid_value = arr.LoadByteOffset<float>(mid * sizeof(float));
        if (value < mid_value) {
            count = count2;
        } else { // try top half
            first = mid + 1;
            count -= count2 + 1;
        }
    }
    return first;
}
