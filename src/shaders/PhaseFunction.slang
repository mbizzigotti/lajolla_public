import Shared;
import Frame;

float3 eval(in PhaseFunction phase_function, float3 dir_in, float3 dir_out)
{
    float inv_4_pi = 1 / (4 * float.getPi());
    switch (phase_function.type)
    {
    case PhaseFunctionType::Isotropic:
    	return inv_4_pi;

    case PhaseFunctionType::HenyeyGreenstein:
    {
        float g = phase_function.param.x;
		return (inv_4_pi * (1 - g * g) /
			(pow((1 + g * g + 2 * g * dot(dir_in, dir_out)), 1.5)));
	}
    }
    return 0;
}

Optional<float3> sample_phase_function(
    in PhaseFunction phase_function, float3 dir_in, float2 rnd_param)
{
    switch (phase_function.type)
    {
    case PhaseFunctionType::Isotropic:
    {
        // Uniform sphere sampling
        float z = 1 - 2 * rnd_param.x;
        float r = sqrt(fmax(0, 1 - z * z));
        float phi = 2 * float.getPi() * rnd_param.y;
        return float3(r * cos(phi), r * sin(phi), z);
	}

    case PhaseFunctionType::HenyeyGreenstein:
    {
        float g = phase_function.param.x;
		// When g ~= 0, this equation degenerates, thus we switch to uniform sphere
		// sampling when that happens
		if (fabs(g) < float(1e-3)) { // the constant was taken from pbrt
			float z = 1 - 2 * rnd_param.x;
            float r = sqrt(fmax(0, 1 - z * z));
            float phi = 2 * float.getPi() * rnd_param.y;
			return float3(r * cos(phi), r * sin(phi), z);
		} else {
			float tmp = (g * g - 1) / (2 * rnd_param.x * g - (g + 1));
			float cos_elevation = (tmp * tmp - (1 + g * g)) / (2 * g);
			float sin_elevation = sqrt(max(1 - cos_elevation * cos_elevation, 0));
			float azimuth = 2 * float.getPi() * rnd_param.y;
			Frame frame = Frame(dir_in);
			return to_world(frame,
				float3(sin_elevation * cos(azimuth),
						sin_elevation * sin(azimuth),
						cos_elevation));
		}
    }
    }
    return none;
}

float pdf_sample_phase(in PhaseFunction phase_function,
                       float3 dir_in, float3 dir_out)
{
    float inv_4_pi = 1 / (4 * float.getPi());
    switch (phase_function.type)
    {
    case PhaseFunctionType::Isotropic:
        return inv_4_pi;

    case PhaseFunctionType::HenyeyGreenstein:
    {
        float g = phase_function.param.x;
    	return inv_4_pi * (1 - g * g) / 
            (pow((1 + g * g + 2 * g * dot(dir_in, dir_out)), 1.5));
    }
    }
    return 0;
}
