#ifndef ENDFIELD_TOON_BASE_BUFFER_INCLUDED
#define ENDFIELD_TOON_BASE_BUFFER_INCLUDED

#define SHADOW_SAMPLER sampler_linear_clamp_compare
SAMPLER_CMP(SHADOW_SAMPLER);

float cmp(bool condition)
{
    return condition ? 1.0f : 0.0f;
}

float2 cmp(bool2 condition)
{
    float x = condition.x ? 1.0f : 0.0f;
    float y = condition.y ? 1.0f : 0.0f;
    return float2(x,y);
}

float3 cmp(bool3 condition)
{
    float x = condition.x ? 1.0f : 0.0f;
    float y = condition.y ? 1.0f : 0.0f;
    float z = condition.z ? 1.0f : 0.0f;
    return float3(x,y,z);
}

float normalizeMinMax(float x, float minVal, float maxVal)
{
    float range = max(maxVal - minVal, 0.0001);
    return (x - minVal) * rcp(range);
}


// region EndField 精简版 params

// #define _cb0_6 float4(0.0600995, -0.0446703, 0.9971923, 0.00)  //dir吗？ xz: , y: 
// #define _cb0_32 float4(0.2486042, 301.1375122, 3.4918606, 0.00)  //xyz: pos吗

#define _cb0_65 float4(-1.00, 0.10, 50.00, 0.02)  //y: 

#define _cb0_66 float4(0.00, 0.00, 0.00, 0.00)  //unity_OrthoParams.w: 这是判断透视相机还是正交相机的参数
#define _cb0_82 float4(13.9273453, 278.5469055, 557.093811, 208.4173279)  //x: rain 

#define _cb0_88 float4(-1.00, 0.50, 1.00, 1.9431806E-41)  //x: SampleBias的bias值
#define _cb0_89 float4(1.00, 0.00, 0.00, 0.00)  //x:_ExposureParams SH一阶颜色的强度值的缩放值 and 后面颜色的除数
#define _cb0_90 float4(0.00, 4.0357396E-41, 6.3316270E-41, 6.8360944E-41)  //y:
#define _cb0_91 float4(0.2877225, 0.2877225, 1.00, 0.00)  //x: _EnvironmentGlobalParams0 SH一阶颜色的强度值

// #define _cb0_131 float4(0.00, 0.00, 0.00, 0.001)  //xyz: _FogColor_E, w:
// #define _cb0_132 float4(0.00, 0.00, 0.00, 0.00)  //xyz: _FogColor_A, w:
// #define _cb0_133 float4(0.00001, 0.00001, 0.00001, -1000.00)  //xyz: _FogColor_B, w:
// #define _cb0_134 float4(0.00, 0.00, 0.00, 0.00)  //xyz: _FogColor_C, w:
// #define _cb0_135 float4(0.00, 0.00, 0.00, 0.00)  //xyz: _FogColor_F, w:
// #define _cb0_136 float4(0.00, 0.00, 1.00, 0.001)  //xyz: _AtmosphereFogParams5, w:
#define _cb0_137 float4(0.00, 0.00, 0.00, 0.00)  //xyz: dir?, w:
#define _cb0_138 float4(0.00, 0.00, 0.00, 0.00)  //xyz: dir?, w:
// #define _cb0_139 float4(0.00, 0.00, 0.00, 1.00)  //xyz: 额外的雾效颜色, w:
#define _cb0_140 float4(0.00, 0.00, 0.00, 0.00)  //xyz: dir?, w:

#define _cb0_141 float4(0.00, 0.00, 0.00, 0.00)  //z: 雾效里特性开关


#define _cb0_157 float4(224.3821564, 86.9738693, -599.5462036, 278.5469055)  //
// #define _cb0_160 float4(1.00, 1.05, 0.55, 0.80)  //z: , w: 
// #define _cb0_161 float4(0.00, 1.00, 1.00, 1.00)  //x: 插值, y: _CharacterParams1.y，z: 插值, w: 插值

#define _cb0_162 float4(0.8783069, 0.9302293, 1.1216931, 0.30)  //xyz: _LaserDefaultColor不采样纹理的默认镭射代替颜色, w:控制r12.xyz的颜色与镭射颜色插值_LaserIntensity
#define _cb0_163 float4(1.18701, 0.9272287, 0.8129899, 1.00)  //w: 插值

// #define _cb0_164 float4(0.1972383, 0.5299193, 0.8247925, 0.00)  //xyz: 像是lightdir?, w: 插值

// #define _cb0_165 float4(1.00, 1.00, 1.00, 1.00)  //xyz: ,w: 插值
#define _cb0_166 float4(0.00, 1.00, 4.3711388E-08, 0.00)  //xyz: 和normalWS dot, w: 插值
// #define _cb0_167 float4(0.15, 1.50, 0.50, 0.40)  //x: , y: , z: , w:
// #define _cb0_168 float4(0.00, 0.00, 0.00, 1.00)  //xyz: rim color, w: 开关
// #define _cb0_169 float4(8.7422777E-08, -1.00, 0.00, 1.00)  //rim 辅助dir xyz:
// #define _cb0_170 float4(0.00, 1.00, 2.00, -100.00)  //x: main intensity, y: , z: posOS scale, w:
#define _cb0_171 float4(1.00, 0.926182, 0.8666667, 0.00)  //w: _CharacterParams11 SH一阶颜色的插值
#define _cb0_175 float4(224.3821564, 86.9738693, -599.5462036, 1.00)  //xyz: _IVParam0, w: feature开关
#define _cb0_176 float4(0.003125, 0.003125, 0.0125, 0.3333333)  //xyz: _IVParam1
#define _cb0_178 float4(-0.0075508, 0.4722373, 0.0121708, 1.0963056)  //xyzw: _IVDefaultSHAr
#define _cb0_179 float4(-0.0075508, 0.4722373, 0.0121708, 1.0963056)  //xyzw: _IVDefaultSHAg
#define _cb0_180 float4(-0.0075508, 0.4722373, 0.0121708, 1.0963056)  //xyzw: _IVDefaultSHAb


#define _cb1_3 float4(0.1562779, 301.004364, 0.0519908, 1.00)   //customlightPos
#define _cb1_4 float4(1000.00, 0.00, 0.00, 5.6051939E-45)   //w: switch rain dir
#define _cb1_5 float4(2.5888008E-40, 2.5709763E-40, 7.2026741E-43, 1.00)   //w: unity_WorldTransformParams.w
#define _cb1_13 float4(0.00, 0.00, 0.00, 0.00)   //

#define _cb2_0 float4(1.4012985E-44, 5.0446745E-42, 4.4841551E-44, 1.1210388E-43)   //w: asint() = 80
#define _cb2_1 float4(6.3058431E-44, 2.8698593E-42, 3.5873241E-42, 2.0178698E-42)   //y:
#define _cb2_2 float4(0.10, 50.00, 1.00, 1.00)   //w:


#define _cb3_0 float4(0.0213893, -0.6427876, -0.765746, 0.00)   //xyz: dir?

#define _cb3_3 float4(1.00, 1.00, 1.00, 1.6243868)   //xyz: dir?


#define _cb4_30 float4(1.00, 2.00, 0.000434, 6400.00)   //x:
#define _cb4_31 float4(0.00, 0.00, 1.00, 0.00)   //x: , z:


#define _cb5_0 float4(0.215, 1.00, 0.00, 1.00)  // w: normalScale
#define _cb5_2 float4(1.1801041E-38, 0.00, 1.1801041E-38, 0.00)  //y: vertical_rampColor与diffuse的插值,  w: normalScale
#define _cb5_5 float4(1.1801041E-38, 0.50, 1.00, 0.95)  //y: shadow albedoTint, z: graAlbedo与albedo的插值
#define _cb5_7 float4(0.00, 0.50, 1.1801041E-38, 0.00)  //x: 使用albedo的alpha _SurfaceType
#define _cb5_9 float4(1.1801041E-38, 0.50, 1.1801041E-38, 0.00)  //
#define _cb5_14 float4(1.1242853E-38, 1.1430113E-38, 9.00, 0.00)    //z: _EmissionIntensity, w: Back Face Normal Flip
#define _cb5_19 float4(0.00, -1000.00, 0.00, 1.00)    //w: 选择燃烧feature使用uv0还是uv1， 用_DECAL_UV关键字替代
#define _cb5_20_cloth2 float4(0.00, 0.00, 1.00, 0.00)    //x: ndv偏移, y: , z: ndv power, w:
#define _cb5_20_cloth3 float4(0.00, 0.00, 1.00, 1.00)    //x: ndv偏移, y: , z: ndv power, w:

#define _cb5_21 float4(1.00, 0.505, 10.00, 1.02)    //x: , y: , z: 
#define _cb5_26 float4(1.00, 1.00, 1.00, 1.00)    //_BaseColor
#define _cb5_27 float4(1.00, 1.00, 1.00, 1.00)    //_EmissionCol
#define _cb5_44 float4(0.00, 0.00, 0.00, 0.00)    //xy: , zw: 
#define _cb5_45 float4(110.1072845, 1.8758985, 0.00, 1.00)    //xyz: 某种颜色吗, w: 
#define _cb5_46 float4(1.00, 0.00, 0.00, 1.00)    //xyz: 颜色, w: 
#define _cb5_47 float4(1.00, 0.017038, 0.00, 1.00)    //
#define _cb5_57 float4(1.00, 1.00, 0.00, 0.00)    //

//cbt-3 body params
//ubo 0 7
#define _17_m28 float4(6.0098567, 120.19713593, 240.39427185, 16.78515625)  //_Time

#define _17_m38 float4(1.00, 0.00, 0.00, 0.00)
#define _17_m40 float4(0.28772247, 0.28772247, 1.00, 0.00)
#define _17_m94 float4(1.00, 1.04999995, 0.55000001, 0.80000001)
#define _17_m95 float4(0.00, 1.00, 1.00, 1.00)
#define _17_m97 float4(1.18701005, 0.92722869, 0.81298989, 1.00)
#define _17_m98 float4(1.00, 0.92618197, 0.86666667, 1.00)
#define _17_m100 float4(0.00, 1.00, 4.37113883E-8, 0.00)
#define _17_m101 float4(0.15000001, 1.50, 0.50, 0.00)
#define _17_m102 float4(0.00, 0.00, 0.00, 1.00)

#define _17_m103 float4(8.74227766E-8, -1.00, 0.00, 0.40000001)
#define _17_m104 float4(0.00, 1.00, 2.00, -100.00)

#define _17_m105 float4(0.10355083, 0.52991927, 0.84170228, 0.00)
#define _17_m106 float4(1.00, 1.00, 1.00, 0.00)
#define _17_m107 float4(0.00, 0.00, 0.00, 1.00)

//ubo 1 0
#define _46_m1 0.454f

#define _46_m30 float4(0.38132611, 0.12477184, 0.13843162, 1.00)
//ubo 2 0
// #define _20_m0_12_m7 float4(0.00, 0.00, 0.00, 0.00)
//ubo 3 12
#define _33_m0 float4(0.02138927, -0.64278764, -0.76574594, 0.00)
#define _33_m3 float4(1.00, 1.00, 1.00, 1.62438679)
//ubo 3 14
#define _35_m6 float4(1.00, 2.00, 0.00043403, 6400.00)

//region old struct
struct DotData
{
    float ndl;
    float ndh;
    float ndv;
    float vdh;
    float ldh;
};

struct VecData
{
    float3 positionOS;
    float3 positionWS;
    float4 positionHCS;
    float3 normalWS_raw;
    float3 tangentWS_raw;
    float3 bitangentWS_raw;
    
    float3 normalWS;
    float3 tangentWS;
    float3 bitangentWS;
    
    float3 lightDirWS;
    float3 viewDirWS;
    float3 halfDirWS;
    float3 reflectDirWS;
};

//region new struct
struct Attributes
{
    float4 vertex       :POSITION;
    float3 normal       :NORMAL;
    float4 tangent      :TANGENT;
    float4 color        :COLOR;
    float2 uv0          :TEXCOORD0;
    #if _CHARACTER_VFX_SPECIAL
    float2 uv1          :TEXCOORD1;     //emissionMap uv1
    float2 uv2          :TEXCOORD2;     //smoothNormal
    float2 uv4          :TEXCOORD3;     //index xy
    float2 uv5          :TEXCOORD4;     //index zw
    float2 uv6          :TEXCOORD5;     //weight xy
    float2 uv7          :TEXCOORD6;     //weight zw
    #else
    float2 uv1          :TEXCOORD1;     //smoothNormal
    float2 uv4          :TEXCOORD2;     //index xy
    float2 uv5          :TEXCOORD3;     //index zw
    float2 uv6          :TEXCOORD4;     //weight xy
    float2 uv7          :TEXCOORD5;     //weight zw
    #endif
    UNITY_VERTEX_INPUT_INSTANCE_ID 
};

struct Varyings 
{
    float4 positionHCS      :SV_POSITION;
    float4 uv               :TEXCOORD0;// xy:uv0 zw:uv1
    float4 normalWS         :TEXCOORD1;
    float4 tangentWS        :TEXCOORD2;
    float4 bitangentWS      :TEXCOORD3;
    float3 positionWS       :TEXCOORD4;
    float4 color            :TEXCOORD5;
    float4 viewDirWS        :TEXCOORD6;
    float3 normalOS         :TEXCOORD7;
    float3 positionOS       :TEXCOORD8;
    // uint   facing           :VFACE;
    
    // Other Props

    UNITY_VERTEX_INPUT_INSTANCE_ID
    UNITY_VERTEX_OUTPUT_STEREO
};

struct EndFieldLightData
{
    float3 position;    //位置
    float3 direction;   //方向
    float3 color;       //颜色
    float  intensity;   //强度
    float  attenuation; //衰减

    float3 standardColor;          //基础材质灯光颜色
    float  useStandardColor;       //使用 基础材质灯光颜色
    float  useStandardIntenstity;  //使用 基础材质灯光强度
};

struct EndFieldVecData
{
    float3 lightDirWS;
    float3 lightDirWS_XZ;
    float3 viewDirWS;
    float3 cameraForwardWS;
    float3 cameraForwardWS_XZ;
    float3 cameraLeftWS;
    float3 halfDirWS;
    float3 reflectDirWS;
};

struct EndFieldDotData
{
    float NdotL;
    float NdotL_XZ;
    
    float NdotH;
    float NdotV;
    float VdotH;
    float LdotH;

    float NdotL_clamped;
    float NdotV_clamped;
    float NdotV2_clamped;

    float NgtLdotForwardWS_clamped;
};

struct EndFieldShadowData
{
    float char;
    float scene;
};

struct EndFieldSurface
{
    float2 baseUV;
    float2 baseUV1;
    float3 basecolor;
    float3 basecolor_shadow;
    float alpha;
    float metallic;
    float specularLevel;
    float anisotropy;
    float perceptualsmoothness;
    float perceptualRoughness;
    float perceptualRoughness_rain;//雨水的粗糙度
    float rainMask;//带有雨水区域mask
    float occlusion;
    float3 emission;
    float  depth;
    
    float3 normalWS_raw;
    float3 tangentWS_raw;
    float3 bitangentWS_raw;
    float  tangent_W;

    float3x3 TBNWS;
    
    float3 normalOS;
    float3 normalWS;
    float3 tangentWS;
    float3 bitangentWS;

    float3 normalWS_withRain;

    float3 positionOS;
    float3 positionWS;
    float4 positionHCS;
    
    float2 screenUV;
    float2 screenPos;

    uint renderingLayerMask;
};

struct EndFieldEnvData
{
    float useEnvFeature;
    float envIntensity;
    float3 envDirWS;
    float3 envColor_raw;
    float3 envColor_fixed;
};

struct EndFieldBSDF
{
    float3 diffuse;
    float3 diffuse_shadow;
    float3 diffuse_shadow2;
    float  specularLevel;
    float3 F0;
    float3 F90;
    float  perceptualRoughness;
    float  roughness;
    float  anisotropy;
    float  roughnessT;
    float  roughnessB;
    float  roughness2;
    float  occlusion;

    float4 ramp_NdotL;
    float ramp_NdotV;
};

struct Lighting
{
    float3 diffuse;
    float3 specular;
};

struct AggregateLighting
{
    Lighting direct;
    Lighting indirect;
};

struct LightLoopOutput
{
    float3 diffuseLighting;
    float3 specularLighting;
};

//region BSDF func
void ApplyToEndFieldEnvData(float envGlobalIntensity, float4 shAr, float4 shAg, float4 shAb, EndFieldSurface surface,
    inout EndFieldEnvData envData)
{
    //询问"从normal方向看，环境光有多亮？"
    float3 sh01 = max(SHEvalLinearL0L1(surface.normalWS.xyz, shAr, shAg, shAb), 0.0f);

    float3 shColor_raw = envGlobalIntensity * sh01.xyz;

    float3x3 SH01_mtx = float3x3(shAr.xyz, shAg.xyz, shAb.xyz);
    float3 shVector_gray = mul(SH01_mtx, float3(0.2126, 0.7152, 0.0722));
    float3 envLightDirection = SafeNormalize(shVector_gray);
    envLightDirection.y = abs(envLightDirection.y);

    //调整SH颜色
    float3 hsv = RgbToHsv(shColor_raw);

    // 调整饱和度和亮度
    float edgeSmooth = (abs(hsv.x - 0.5f) - 0.45f) * -10.0f;
    float smoothCurve = smoothstep(0, 1, edgeSmooth);

    hsv.y = min(hsv.y, (0.70f - 0.35f * smoothCurve) * saturate(hsv.z));
    hsv.z = 2.0f / (2.0f - hsv.y);

    envData.useEnvFeature = 1.0f;
    envData.envColor_raw = shColor_raw;
    envData.envColor_fixed = HsvToRgb(hsv);
    envData.envDirWS = envLightDirection;

    //询问"从lightDir方向看，环境光有多亮？",得到的就是环境光的最大亮度
    float3 envLightMaxColor = max(SHEvalLinearL0L1(envLightDirection, shAr, shAg, shAb), 0.0f);
    float envLightMaxIntensity = Max3(envLightMaxColor.x, envLightMaxColor.y, envLightMaxColor.z);

    envData.envIntensity = envGlobalIntensity * envLightMaxIntensity;
}

float3 CalculatePureLaserColor(float3 RGB)
{
    float3 hsv = RgbToHsv(RGB);
    float H = hsv.x;  // 色相
    float S = hsv.y;  // 饱和度

    // Step 2: 生成纯彩虹色（饱和度=1，亮度=1的标准彩虹）
    float3 pureRainbow = HsvToRgb(float3(H, 1.0, 1.0));

    // - 基于饱和度在白色(1,1,1)和彩虹色之间插值
    // - 饱和度越高，亮度越强（S=1时放大2倍）
    float brightnessBoost = 2.0 / (2.0 - S);
    float3 boostedColor = brightnessBoost * lerp(1.0, pureRainbow, S);

    return boostedColor;
}

void ApplyEnvFeature(EndFieldSurface surface,
    inout EndFieldEnvData envData)
{
    //_CharacterParams11.w = 1时，不开启雾效
    float envGlobalIntensity = lerp(_EnvironmentGlobalParams0.x, 1, _CharacterParams11.w) * _ExposureParams.x;
    float3 envColor = 0;
    if (_CharacterParams1.y < 0.5)
    {
        float3 lightBoxPos = _IVParam0.xyz;
        float useLightBox = _IVParam0.w;
        float3 lightBoxDir = surface.positionWS.xyz - lightBoxPos;
        float maxLightBoxDistance = Max3(abs(lightBoxDir.z), abs(lightBoxDir.y), abs(lightBoxDir.x));
        //当前存在于lightbox中的系数，0不在，1在中心
        float indirRange = saturate((maxLightBoxDistance - 896.0f) * 0.015625f); //0.015625f = 1 / 64

        float4 shAr, shAg, shAb;

        float cmp2 = cmp(indirRange < 1.0f);
        float cmp3 = 0.0f < useLightBox ? cmp2 : 0;
        //TODO: 以下可能是LPPV，还待优化
        // ===================== tex3 =====================
        if (cmp3 != 0)
        {
            //分两层精度，可能是3dtex 在在同一个lightbox中 有多层精度的
            bool inRange01 = saturate((maxLightBoxDistance - 100.0f) * 0.0833333f) < 1.0f;
            bool inRange02 = saturate((maxLightBoxDistance - 200.0f) * 0.0625f) < 1.0f;
            //0.00390625f = 1 /256, 0.001953125f = 1 / 512, 0.0004882813f = 1 / 2048
            float posSize = inRange01 ? 0.00390625f : (inRange02 ? 0.001953125f : 0.0004882813f);
            float3 T3UV = float3(frac(posSize * surface.positionWS.xyz));
            float4 tex3d_3 = _T3.SampleLevel(sampler_T3, T3UV, inRange01 ? 0.0f : (inRange02 ? 1.0f : 2.0f));
            float hasLightBox = floor(tex3d_3.w * 255.0f + 0.5f);

            // ===================== tex4、5 =====================
            if (0.0f < hasLightBox)
            {
                //0.00312 = 1 / 320, 0.0125 = 1/ 80
                float3 tex3d_Size = _IVParam1.xyz;
                float3 punctualLightShadowTexV2_UVW = frac(surface.positionWS.xyz / hasLightBox);
                punctualLightShadowTexV2_UVW = punctualLightShadowTexV2_UVW * 4 + 0.5;
                punctualLightShadowTexV2_UVW += floor(mad(tex3d_3.x, 255.0f, 0.5f)) * 5;
                punctualLightShadowTexV2_UVW *= tex3d_Size;

                float4 pLightShadow00 = _T4.SampleLevel(sampler_T3, punctualLightShadowTexV2_UVW, 0.0f);
                
                float4 pLightShadow01 = _T5.SampleLevel(sampler_T3, float3(punctualLightShadowTexV2_UVW.xy, punctualLightShadowTexV2_UVW.z * 0.3333333f), 0.0f);
                float4 pLightShadow02 = _T5.SampleLevel(sampler_T3, float3(punctualLightShadowTexV2_UVW.xy, punctualLightShadowTexV2_UVW.z * 0.3333333f + 0.3333333f), 0.0f);
                float4 pLightShadow03 = _T5.SampleLevel(sampler_T3, float3(punctualLightShadowTexV2_UVW.xy, punctualLightShadowTexV2_UVW.z * 0.3333333f + 0.6666667f), 0.0f);

                float3 _81_890 = pLightShadow00.x * (pLightShadow01.xyz * 4.0f - 2.0f);
                float3 _82_567 = pLightShadow00.y * (pLightShadow02.xyz * 4.0f - 2.0f);
                float3 _83_123 = pLightShadow00.z * (pLightShadow03.xyz * 4.0f - 2.0f);

                float4 ar = float4(_81_890, pLightShadow00.x);
                float4 ag = float4(_82_567, pLightShadow00.y);
                float4 ab = float4(_83_123, pLightShadow00.z);

                shAr = lerp(ar, _IVDefaultSHAr, indirRange);
                shAg = lerp(ag, _IVDefaultSHAg, indirRange);
                shAb = lerp(ab, _IVDefaultSHAb, indirRange);
            }
            else
            {
                shAr = _IVDefaultSHAr;
                shAg = _IVDefaultSHAg;
                shAb = _IVDefaultSHAb;
            }
        }
        else
        {
            shAr = _IVDefaultSHAr;
            shAg = _IVDefaultSHAg;
            shAb = _IVDefaultSHAb;
        }

        shAr = _AmbientProbeData[0];   //test code
        shAg = _AmbientProbeData[1];   //test code
        shAb = _AmbientProbeData[2];   //test code

        ApplyToEndFieldEnvData(envGlobalIntensity, shAr, shAg, shAb, surface, envData);
    }
    else
    {
        // ===================== 属于是不使用sh color来附加一下变化的保底做法（副分支：cb0[161].y ≥ 0.5） =====================
        //采样t15 matcap做法的镭射
        if (1.5f < _CharacterParams1.y)
        {
            float3 laserNormalVS = TransformWorldToViewNormal(surface.normalWS, true);
            float4 laserMap = SAMPLE_TEXTURE2D(_LaserMap, sampler_LaserMap, laserNormalVS.xy * 0.5 + 0.5);

            float3 laser = CalculatePureLaserColor(laserMap.xyz);
            envColor = lerp(1.0f, laser, _CharacterParams2.w);    //强度控制
        }
        else
        {
            float3 standardEnvColor = _CharacterParams2.xyz;
            envColor.xyz = standardEnvColor;     //_CharacterParams2
        }

        envData.envColor_fixed = envColor;
        envData.envColor_raw = 1.0f;
        envData.envDirWS = 0.0f;
        envData.useEnvFeature = 0.0f;       
        envData.envIntensity = envGlobalIntensity;
    }
}

float4 SampleRampColor_NdotL(EndFieldVecData vecData, EndFieldDotData dotData)
{
    // 视角修正 (View Dependent Bias)
    float viewPitchBias = smoothstep(0, 1, (0.75f - abs(vecData.cameraForwardWS.y)) * 2.0f);   //视角保持水平为1，其余角度为0
    float LdotCamForDir = dotData.NgtLdotForwardWS_clamped;   //负光照方向 dot 视线方向，等于在看向光源位置是1，其余是0

    float backFactor = viewPitchBias * LdotCamForDir;
    float NdotL_Bias = dotData.NdotL * (-dotData.NdotL * 0.5) + 0.5f;

    // 给NdotL的背面（负数区域）加上光照效果, 视角在光照背面时，将明暗交界线推移到后方，有更多细节
    float NdotL_Fixed = dotData.NdotL + lerp(backFactor * NdotL_Bias, _CharacterParams4.w, _CharacterParams3.w);    //启用的开关，是否有更多细节，是否简化细节
    NdotL_Fixed = clamp(NdotL_Fixed, -1, 1);

    float2 rampUV;
    rampUV.x = NdotL_Fixed * 0.5 + 0.5;
    rampUV.y = 0.5;
    float4 rampColor = SAMPLE_TEXTURE2D_LOD(_ShadowRampTex, sampler_ShadowRampTex, rampUV, 0);      //控制颜色变化的ramp
    
    return rampColor;
}

float SampleRampColor_NdotV(EndFieldVecData vecData, EndFieldSurface surface)
{
    float3 cameraForwardWS_fixed = float3(vecData.cameraForwardWS.x,
                                    min(vecData.cameraForwardWS.y + 0.25f, 1.0f),
                                    vecData.cameraForwardWS.z);
    float3 cameraOffsetDir_nlz = normalize(cameraForwardWS_fixed);
    float2 rampUV_alpha;
    rampUV_alpha.x = dot(surface.normalWS, cameraOffsetDir_nlz) * 0.5 + 0.5;    //NdotV的微调版
    rampUV_alpha.y = 1.0f;
    float rampAlpha = SAMPLE_TEXTURE2D_LOD(_ShadowRampTex, sampler_ShadowRampTex, rampUV_alpha, 0).a;   //控制亮暗渐变程度的ramp
    return rampAlpha;
}

float D_EndField(EndFieldDotData dotData, EndFieldBSDF bsdf)
{
    float rain_NdotH2 = dotData.NdotH * dotData.NdotH;
    float rough2 = bsdf.roughness * bsdf.roughness;
    float deom = rain_NdotH2 * (rough2 - 1.0f) + 1.0f;
    float deom2 = deom * deom;
    float ggxRaw = (rough2 != deom2) ? rough2 / deom2 : 1.0f;
    
    //随视角变化，边缘会有不同
    float viewDependentFactor = dotData.NdotV_clamped * 2 + bsdf.roughness + 0.00001f;        //r1.y: roughness
    float specular = ggxRaw * rcp(viewDependentFactor);
    specular = specular * 0.5 - 6.10351563e-005;
    specular = clamp(specular, 0, 20);
    return specular;
}

float3 F_EndField(EndFieldDotData dotData, EndFieldSurface surface, EndFieldBSDF bsdf)
{
    float rain_NdotH2 = dotData.NdotH * dotData.NdotH;
    float rough2 = bsdf.roughness * bsdf.roughness;
    float deom = rain_NdotH2 * (rough2 - 1.0f) + 1.0f;
    float deom2 = deom * deom;
    float ggxRaw = (rough2 != deom2) ? rough2 / deom2 : 1.0f;
    float ggx = ggxRaw * (rough2 + 9.99999975e-005);     //保证有最小高光

    float2 specRampUV;
    specRampUV.x = lerp(ggx, dotData.NdotV2_clamped, _SpecRampIridescentMode);
    specRampUV.y = surface.perceptualRoughness * (1.0f - surface.metallic);     //这里用不带雨水的感知粗糙度
    float3 specRampColor = SAMPLE_TEXTURE2D_LOD(_SpecRampMap, sampler_SpecRampMap, specRampUV, 0).xyz;
    float3 F = bsdf.F0 * specRampColor;
    return F;
}

//应用下雨淋湿材质的feature
void ApplyRainFeature(inout EndFieldSurface surface)
{
    float rainStrengthCombined = lerp(_cb1_13.x + _RainEffectIntensity,       _CharacterParams10.y, _CharacterParams10.x);    //主控制强度的变化
    float wetHeightCombined    = lerp(_cb1_13.z + _WetEffectWorldSpaceHeight, _CharacterParams10.w, _CharacterParams10.x);    //可能是水面深度的控制
    float wetIntensityCombined = lerp(_cb1_13.y + _WetEffectIntensity,                           1, _CharacterParams10.x);    //不清楚

    // 2. 计算高度遮罩 (Height Mask)
    float heightMask = smoothstep(0, 1, 2.85714269 * (wetHeightCombined - surface.positionWS.y + 0.2f));

    float wetnessFactor = heightMask * wetIntensityCombined;                  //这两个是一对，1163是另一个模块
    float totalRainFactor = heightMask * wetIntensityCombined + rainStrengthCombined ;  //其中heightMask * wetIntensityCombined模块是控制水深度的

    if (9.99999975e-005 < totalRainFactor)
    {
        float inv_metallic              = 1.0f - surface.metallic;
        // 雨水在非金属表面更明显，计算一个基于亮度的遮罩，用于后续变色
        float rain_diffuse_gray         = Luminance(surface.basecolor.xyz * inv_metallic);
        float rain_diffuse_smt          = smoothstep(0.0f, 1.0f, (rain_diffuse_gray - 0.35f) * -4.0f);// 暗部更容易显得湿

        bool filpObjectDir = _RainFilpObjectDir != 0.0f;    //切换流向

        float3 posOS = filpObjectDir ? surface.positionOS.xzy * float3(1.0f, 1.0f, -1.0f) : surface.positionOS.xyz;
        float3 posOS_scale = posOS.xyz * _CharacterParams10.z;

        float3 normalOS = filpObjectDir ? surface.normalOS.xyz : surface.normalOS.xzy;
        float3 absNormal = abs(normalOS) - 0.2f;

        float3 triWeights = max(absNormal.xzy * absNormal.xzy * absNormal.xzy, 0.0f);
        triWeights /= (triWeights.x + triWeights.y + triWeights.z + 1e-5f);
        // ================= 采样点状雨 (Point Rain Map) =================
        float4 pointRainMap1 = SAMPLE_TEXTURE2D(_CharacterRainEffectTex, sampler_CharacterRainEffectTex, posOS_scale.xz);
        float4 pointRainMap2 = SAMPLE_TEXTURE2D(_CharacterRainEffectTex, sampler_CharacterRainEffectTex, posOS_scale.xy);   // Top projection
        float4 pointRainMap3 = SAMPLE_TEXTURE2D(_CharacterRainEffectTex, sampler_CharacterRainEffectTex, posOS_scale.zy);

        float pointNormalX        = dot(float3(pointRainMap3.x, pointRainMap1.x, pointRainMap2.x), triWeights);
        float pointNormalY        = dot(float3(pointRainMap3.y, pointRainMap1.y, pointRainMap2.y), triWeights);
        float pointMask           = dot(float3(pointRainMap3.z, pointRainMap1.z, pointRainMap2.z), triWeights);
        float pointWet            = dot(float3(pointRainMap3.w, pointRainMap1.w, pointRainMap2.w), triWeights);//全部的痕迹


        // 计算湿润痕迹的强弱 (Based on Normal Y and Metallic)
        // 垂直表面(Normal Y小)或者金属表面，水珠效果不同
        float wetTraceBias = saturate(rainStrengthCombined * inv_metallic + surface.normalWS.y * 0.2f);

        // 两种阈值的湿润区域：大范围(Most)和小范围(Less)
        float mixWet_most = ((pointWet - 0.80f) + wetTraceBias) * 3.33333f;//一大半的痕迹
        float mixWet_less = ((pointWet - 0.45f) + saturate(wetnessFactor * inv_metallic)) * 1.538f;//一小些的痕迹

        float pointWetMixMask = max(smoothstep(0.0f, 1.0f, mixWet_most), smoothstep(0.0f, 1.0f, mixWet_less));
        float rainMaxParam    = max(wetnessFactor, rainStrengthCombined);

        // 计算雨点密度遮罩 (Droplets Mask)
        // 金属度越高或越光滑，水珠越明显
        float wetMetallic_smt            = smoothstep(0.0f, 1.0f, (surface.metallic - 0.5f) * 4.0f);       //raining metallic
        float wetPerceptualRoughness_smt = smoothstep(0.0f, 1.0f, (surface.perceptualRoughness - 0.3f) * -10.0f);
        float dropletThreshold           = min(wetPerceptualRoughness_smt + wetMetallic_smt, 1.0f);

        // 最终的点状雨强度
        float dropletsIntensity          = (dropletThreshold * rainMaxParam - (1.0f - pointMask)) * 10.0f;    //结果是金属度越高，点越多；粗糙度越低，点越多

        // ================= 采样流状雨 (Vertical Rain Map) =================
        float flowSpeedU = 0.0f;
        float flowSpeedV = _Time.x * _CharacterParams10.z * 0.75f;
        // 垂直流动的 Triplanar 权重 (Vertical Triplanar)
        // 侧面投影主要受 Normal.XY 影响
        float invLen = rsqrt(max(dot(normalOS.xy, normalOS.xy), 1.1754943508222875079687365372222e-38f));      //safe length inv
        float2 vertWeightsRaw = abs(invLen * normalOS.xy) - 0.20f;

        float2 vertWeights = max(vertWeightsRaw * vertWeightsRaw * vertWeightsRaw, 0.0f);
        vertWeights /= (vertWeights.x + vertWeights.y + 1e-5f);

        float4 streakMap1 = SAMPLE_TEXTURE2D(_CharacterRainStreakTex, sampler_CharacterRainStreakTex, posOS_scale.xy);
        float4 streakMap2 = SAMPLE_TEXTURE2D(_CharacterRainStreakTex, sampler_CharacterRainStreakTex, posOS_scale.zy);

        // 采样流动的 Alpha 通道
        float streakFlow1 = SAMPLE_TEXTURE2D(_CharacterRainStreakTex, sampler_CharacterRainStreakTex, float2(posOS_scale.x + flowSpeedU, posOS_scale.y + flowSpeedV)).w;
        float streakFlow2 = SAMPLE_TEXTURE2D(_CharacterRainStreakTex, sampler_CharacterRainStreakTex, float2(posOS_scale.z + flowSpeedU, posOS_scale.y + flowSpeedV)).w;
        float streakFlowMixed = vertWeights.y * streakFlow1 + streakFlow2 * vertWeights.x;

        // 混合垂直法线
        float2 mixVertNormal = streakMap1.xy * vertWeights.y + streakMap2.xy * vertWeights.x;   //有疑问，翻转了


        // 组合 Normal Map (Point XY + Vertical XY with Flow)
        float rainNormalU = mad(mixVertNormal.x * 2.0f - 1.0f, streakFlowMixed, pointNormalX * 2.0f - 1.0f);
        float rainNormalV = mad(mixVertNormal.y * 2.0f - 1.0f, streakFlowMixed, pointNormalY * 2.0f - 1.0f);

        // 计算流痕强度 (Streak Intensity)
        float streakBase = streakMap1.z * vertWeights.y + streakMap2.z * vertWeights.x;
        float streakFactor = (dropletThreshold * rainMaxParam - (1.0f - streakBase)) * 10.0f;

        // 最终雨水法线强度混合 (0~1)
        float finalRainNormalStrength = max(smoothstep(0.0f, 1.0f, streakFactor), smoothstep(0.0f, 1.0f, dropletsIntensity));    //受粗糙度、金属度、开关控制的强度


        // ================= 3. 构建切线空间与混合法线 =================
        // 1. 构建切线空间基底 (Tangent Basis Construction)
        // 逻辑相当于 Cross(Normal, (0,1,0))，构建一个水平切线
        float3 N = surface.normalWS.xyz;
        float3 upVector = float3(0, 1, 0);

        // 计算未归一化的切线 T_raw = (-N.z, 0, N.x)
        float3 T_raw = float3(-N.z, 0.0f, N.x); 

        // 计算长度平方，检查是否产生奇点 (Singularity Check)
        float T_lenSq = dot(T_raw, T_raw);
        bool  noSingularity = T_lenSq > 6.1035e-05f;
        float T_invLen = rsqrt(T_lenSq);

        // 归一化切线 T
        // 如果无奇点：T = -normalize(T_raw)
        // 如果有奇点(法线朝上)：T = (-1, 0, 0)
        float3 T_axis = noSingularity ? -T_invLen * T_raw : float3(-1.0f, 0.0f, 0.0f);

        // 2. 计算副切线 B
        float3 B_axis = cross(N, T_axis);

        // 3. 应用雨水扰动 (Perturbation using Lerp Chain)
        // 这是一个非常特殊的混合方式，不是 N + T*u + B*v，而是连续插值

        // 第一步：在 N 和 T 之间插值
        float3 dir_step1 = lerp(N, T_axis, rainNormalU);

        // 第二步：在 上一步结果 和 B 之间插值
        float3 dir_step2 = lerp(dir_step1, B_axis, rainNormalV);

        // 4. 归一化雨水法线
        float3 waterNormal_nlz = normalize(dir_step2);

        // 混合 原始法线 和 雨水法线
        float3 mixNormalWS = lerp(N, waterNormal_nlz, finalRainNormalStrength); //用受粗糙度、金属度、开关控制的强度，混合原始法线和雨水法线
        float3 mixNormalWS_nlz = normalize(mixNormalWS);


        // ================= 4. 修改材质属性 (Color/Roughness) =================
        // 粗糙度处理：水面非常光滑 (0.05)
        // 混合原始粗糙度和水面粗糙度
        float waterPerceptualRoughness = min(surface.perceptualRoughness, 0.05f);
        float mixPerceptualRoughness = lerp(surface.perceptualRoughness, waterPerceptualRoughness, finalRainNormalStrength);  //混合原本的粗糙度和雨水沾湿的粗糙度
        // 在湿润遮罩(pointWetMixMask)区域，进一步降低粗糙度，但保留底限 0.2
        float finalMixPerceptualRoughness = max((mixPerceptualRoughness - rain_diffuse_smt * pointWetMixMask * 0.20f), min(mixPerceptualRoughness, 0.20f));//在混合粗糙度里，降低湿润区域的粗糙度，后面的min(0.2f)是保留最低的雨水粗糙度

        // 颜色处理：变暗 (Darkening)
        // 计算基于亮度的变暗系数 (Wet surfaces look darker)
        float darkenBase   = (Luminance(surface.basecolor) - 0.7f) * -2.5f;
        float darkenFactor = smoothstep(0.0f, 1.0f, darkenBase) * 0.5f + 1.0f;   //淋湿区域的albedo强度系数, 1.0 ~ 1.5 倍

        // 混合系数
        float rainAlbedoMix = wetMetallic_smt * finalRainNormalStrength; //雨水会挂在金属上，所以有金属度系数
        float wetAlbedoMix  = 1.0f - 0.5f * (1.0f - wetPerceptualRoughness_smt) * (1.0f - rain_diffuse_smt) * pointWetMixMask;//像是整体控制湿润强度的

        surface.perceptualRoughness_rain = waterPerceptualRoughness;  //雨水沾湿区域的粗糙度
        
        // 应用变暗
        surface.basecolor = lerp(surface.basecolor, surface.basecolor * darkenFactor, rainAlbedoMix) * wetAlbedoMix;//亮面albedo，有雨水变色、淋湿变色。淋湿区域，深颜色会变浅，浅颜色会变深
        surface.basecolor_shadow = surface.basecolor_shadow * wetAlbedoMix;   //阴影albedo只受整体湿润影响，因为在暗面不受雨水效果

        surface.perceptualRoughness = finalMixPerceptualRoughness;//带雨水、淋湿的粗糙度

        surface.normalWS_withRain = mixNormalWS_nlz;  //混合了雨水的法线
        surface.rainMask = finalRainNormalStrength;//受粗糙度、金属度、开关控制的强度
    }
}

//用用特殊效果feature
float3 ApplyVFXFeature(EndFieldVecData vecData, EndFieldSurface surface)
{
    #ifdef _CHARACTER_VFX_SPECIAL
        // 计算基础透明度参考值
        float globalAlphaRef = _VFXColor.w * _VFXColorAlpha;
        float2 MapUV = _VFXMainUVSet > 0.5f ? surface.baseUV1 : surface.baseUV;
        
        float2 VFXMainUV = MapUV + _VFXParams0.w * _VFXSpecialParam.zw;
        VFXMainUV = TRANSFORM_TEX(VFXMainUV, _VFXSpecialMainTex);
        float4 VFXMainMap = SAMPLE_TEXTURE2D(_VFXSpecialMainTex, sampler_VFXSpecialMainTex, VFXMainUV);

        float2 VFXBlendUV = VFXMainMap.xx + MapUV;
        VFXBlendUV = VFXBlendUV + _VFXParams0.w * _VFXSpecialParam.xy;
        VFXBlendUV = TRANSFORM_TEX(VFXBlendUV, _VFXSpecialBlendTex);
        float4 VFXBlendMap = SAMPLE_TEXTURE2D(_VFXSpecialBlendTex, sampler_VFXSpecialBlendTex, VFXBlendUV);
        
        // 通道选择: 根据 .x 参数在 Flow图的 A通道 和 R通道 之间插值
        float flowChannel = lerp(VFXBlendMap.w, VFXBlendMap.x, _UseVFXMainTexAsAlpha);
        // 计算不透明度遮罩 (Opacity Mask)
        // 逻辑: (全局Alpha * 流动值 + 主图Alpha) * 混合Tint的Alpha
        float opacityMask = saturate((globalAlphaRef * flowChannel + VFXMainMap.w) * _VFXBlendTint.w);

        // 计算混合图颜色基底
        float3 flowBaseColor = lerp(1.0f, VFXBlendMap.xyz, _UseVFXMainTexAsAlpha);
        // 合成基础火焰颜色 (baseFireColor Base Color)
        // Part A: 主图RGB * 不透明度 * TintColor
        // Part B: 流动图RGB * VFX主色 * 强度乘数(.z)
        float3 baseFireColor = VFXMainMap.xyz * opacityMask * _VFXBlendTint.xyz 
                            + flowBaseColor.xyz * _VFXColor.xyz * _VFXColorIntensity;

        // 计算溶解阈值 (Mapping 0..1 to -1.01..1.02)
        float dissolveThreshold = _SpecialDissolveScheduleOffset * 2.02f - 1.01f;
        // 计算溶解边缘强度 (Edge Factor)
        // 注意: 这里计算的是 "阈值 - 噪声值"。
        float dissolveEdgeFactor = saturate(dissolveThreshold - VFXMainMap.x);

        // 叠加边缘发光色
        // 如果处于溶解边缘内部，叠加 _VFXFresnelColor 作为高亮
        float3 fireColorWithEdge = lerp(baseFireColor, dissolveEdgeFactor * _VFXFresnelColor.xyz + _VFXColorIntensity, dissolveEdgeFactor);

        // 计算视线与法线的点积 (Safe NdotV)
        float VdotN_safe = saturate(dot(vecData.viewDirWS, surface.normalWS_raw/* 这里原本是normalWS_raw_safe ，原始法线的SafeNormalize版本*/) + _VFXFresnelBias);
        VdotN_safe = pow(VdotN_safe, _VFXFresnelPower);
        
        // 菲涅尔反转控制 (Flip)
        float vfxFresnel = lerp(1 - VdotN_safe, VdotN_safe, _VFXFresnelFlip);
        // 计算最终流动遮罩 (Flow Mask)
        float visibilityMask = lerp(1.0f, vfxFresnel, _VFXFresnelAffectOpacity);    //菲涅尔是否影响透明度
        float dissolveCutout = saturate(VFXMainMap.x - dissolveThreshold);
        float flowMask = visibilityMask * saturate(flowChannel * globalAlphaRef * dissolveCutout);
        
        float3 vfxFinalColor = lerp(fireColorWithEdge, _VFXFresnelColor.xyz, vfxFresnel * _VFXFresnelColor.w) * flowMask;  //r10.xyz 熔岩颜色
        float3 effectColor = surface.emission + vfxFinalColor;
    #else
        float3 effectColor = surface.emission;
    #endif
    return effectColor;
}

//手动放置rim灯光位置的feature
float3 ApplyManualRimFeature(float3 diffuseSat, EndFieldVecData vecData, EndFieldShadowData shadowData, EndFieldSurface surface)
{
    float3 rimLightPosWS = UNITY_MATRIX_M._m03_m13_m23; //_RimLightPosWS
    float3 customDirWS;
    customDirWS.xz = surface.positionWS.xz - rimLightPosWS.xz;
    customDirWS.y = 0;
    customDirWS = SafeNormalize(customDirWS);
    
    bool enable_Rim = cmp(0.00999999978 < _CharacterParams8.w);
    //rim
    float VdotN = -abs(dot(vecData.viewDirWS, surface.normalWS));
    float inv_VdotN = VdotN + 1.0f;

    float _2436 = 0.8f - _CharacterParams7.w * 0.6f;
    float _2437 = 0.9f - _CharacterParams7.w * 0.40f;
    float _2447 = 1.0f / (_2437 - _2436) * (inv_VdotN - _2436);
    float rim_VdotN_smt = smoothstep(0, 1, _2447);

    //dot distance
    float rim_atten = saturate(1 + dot(customDirWS.xz, vecData.cameraLeftWS.xz));
    rim_atten = Min3(rim_atten, surface.occlusion, shadowData.char);

    float NdotRimDir_clamp = saturate(dot(vecData.cameraLeftWS, surface.normalWS));

    //mix rim color
    float3 rimDiffuse = lerp(0.25, diffuseSat, _CharacterParams6.w) * NdotRimDir_clamp;
    float3 rimColor = rim_atten * rim_VdotN_smt * _CharacterParams8.xyz * _CharacterParams8.w;      //_cb0_168.xyz: rim color, _cb0_168.w:rim intensity
    float3 rimFinalColor = rimDiffuse * rimColor;
    rimFinalColor = enable_Rim ? rimFinalColor : 0;
    return rimFinalColor;
}

//根据环境球谐自动添加rim的feature
float3 ApplySHRimFeature(float3 diffuseSat, float3 saturatedDirLightColor, EndFieldVecData vecData, EndFieldShadowData shadowData, EndFieldDotData dotData,
    EndFieldEnvData envData, EndFieldSurface surface, EndFieldBSDF bsdf)
{
    float SHdotN = dot(envData.envDirWS, surface.normalWS) * envData.useEnvFeature;

    // 视线与法线夹角的边缘蒙版 (Fresnel-like Edge Mask)
    float VdotN = -abs(dot(vecData.viewDirWS, surface.normalWS));
    float edgeFresnelMask = smoothstep(0, 1, 5.0f * (VdotN + 0.40f));

    // 背光强度因子 (Backlight Factor)
    // 在场景阴影中(shadowData.scene=0)倾向于使用1
    // 在亮部(shadowData.scene=1)倾向于使用负灯光与视角的dot，也就是看向主光方向（逆光）时才是1
    float backlightFactor = lerp(1, dotData.NgtLdotForwardWS_clamped, shadowData.scene);
    
    // 法线与光照方向的匹配蒙版 (Normal-Light Alignment)
    // 在场景阴影中(shadowData.scene=0)倾向于使用环境光方向的dot
    // 在亮部(shadowData.scene=1)倾向于使用主光的dot
    float lightRangeDot = 0.5 - dotData.NdotL_XZ * (dotData.NdotL_XZ * 0.5 - 1);    //拟合曲线
    float rimLightDirectionMask = saturate(lerp(SHdotN, lightRangeDot, shadowData.scene));
    // return float4(SHdotN.xxx, 1);
    
    // 固有色亮度蒙版 (Albedo Brightness Mask)
    // 在场景阴影中(shadowData.scene=0)倾向于使用1
    // 在亮部(shadowData.scene=1)倾向于使用钳制的颜色权重曲线， 防止边缘光出现在本来就很黑的材质上（这会让黑色看起来发灰）
    float albedoBrightnessMask = smoothstep(0, 1, -16.666666 * (Luminance(bsdf.diffuse) - 0.1f));
    albedoBrightnessMask = lerp(1, albedoBrightnessMask, shadowData.scene);
    
    // [Color] 边缘光颜色源 (Rim Source Color)
    // 在场景阴影中(shadowData.scene=0)倾向于使用环境光(SHavgColor)
    // 在亮部(shadowData.scene=1)倾向于使用主光(directionalLightColor)
    float SHavgValue = max(Max3(envData.envColor_raw.x, envData.envColor_raw.y, envData.envColor_raw.z) * 0.5f, 1.0f);
    float3 SHavgColor = envData.envColor_raw * rcp(SHavgValue);
    float3 rimSourceColor = lerp(SHavgColor, saturatedDirLightColor, shadowData.scene);

    float shRimCtrlWeight = 1.0f - _CharacterParams3.w;
    // 综合边缘光强度 (Final Rim Intensity)
    float SHRimIntensity = albedoBrightnessMask * min(shadowData.char, surface.occlusion)
                             * edgeFresnelMask * rimLightDirectionMask * backlightFactor
                             * shRimCtrlWeight;

    return SHRimIntensity * rimSourceColor * max(0.15, diffuseSat);
}
//endregion

//region 初始化数据
void InitializeEndFieldLightData(Light light,
    inout EndFieldLightData lightData)
{

    lightData.standardColor = _CharacterParams5.xyz;
    lightData.useStandardColor = _CharacterParams5.w;
    lightData.useStandardIntenstity = _CharacterParams11.w;
    lightData.direction = light.direction;
    lightData.attenuation = light.distanceAttenuation;  //EndField里是1.6243868
    lightData.intensity = lerp(lightData.attenuation, 1.0f, lightData.useStandardIntenstity);
    lightData.color = lerp(light.color, lightData.standardColor, lightData.useStandardColor) * lightData.intensity;
}

void IntializeEndFieldShadow(EndFieldSurface surface,
    inout EndFieldShadowData shadowData)
{
    float shadowAttenuation_scene = 1;
    float shadowAttenuation_char = 1;

    // 角色的屏幕空间自阴影
    #ifdef _RAYTRACING_SHADOWS
    shadowAttenuation_scene = SAMPLE_TEXTURE2D(_ScreenSpaceShadowmapTexture, sampler_PointClamp, surface.screenUV).x;
    #else
    shadowAttenuation_scene = SAMPLE_TEXTURE2D(_ScreenSpaceShadowmapTexture, sampler_PointClamp, surface.screenUV).x;
    #ifdef _PEROBJECT_SCREEN_SPACE_SHADOW
    shadowAttenuation_char = SamplePerObjectScreenSpaceShadowmap(surface.screenUV);
    // shadowAttenuation_char = min(shadowAttenuation_scene, SamplePerObjectScreenSpaceShadowmap(surface.screenUV));
    #endif
    #endif

    float enableSceneShadow = _DirectionalShadowParams2.x;
    float defaultSceneShadow = _DirectionalShadowParams2.z;
    float sceneShadowIntensity = _DirectionalShadowParams.x;
    float sceneShadowIntensity_volume = _CharacterParams1.z;

    // 场景的投影
    // shadowAttenuation_scene = sampleTex.x;
    float shadowMaskVal = 0.0f < enableSceneShadow ? shadowAttenuation_scene : defaultSceneShadow;
    float shadowMaskMixed = lerp(1, shadowMaskVal, sceneShadowIntensity);
    float sceneShadowAtten = lerp(shadowMaskMixed, 1, sceneShadowIntensity_volume);
    shadowData.char = shadowAttenuation_char;
    shadowData.scene = sceneShadowAtten;
    shadowData.scene = shadowAttenuation_scene;     //test code
    // shadowData.scene = step(0.5, surface.screenUV.y);    //test code
}

void InitializeEndFieldVecData(Varyings i, EndFieldLightData lightData, EndFieldShadowData shadowData, EndFieldSurface surface,
    inout EndFieldVecData vecData)
{
    float  useCustomLight = _CharacterParams1.w;
    float3 customLightDir = _CharacterParams4.xyz;
    vecData.lightDirWS = lerp(normalize(lightData.direction), customLightDir, useCustomLight);
    vecData.lightDirWS_XZ = SafeNormalize(float3(vecData.lightDirWS.x, 0, vecData.lightDirWS.z));
    vecData.viewDirWS = SafeNormalize(i.viewDirWS.xyz);
    vecData.cameraForwardWS = UNITY_MATRIX_I_V._m02_m12_m22;//UNITY_MATRIX_I_V(会同步scene的矩阵)，但_InvViewMatrix只与游戏相机相关
    vecData.cameraForwardWS_XZ = normalize(float3(vecData.cameraForwardWS.x, 0, vecData.cameraForwardWS.z));
    vecData.cameraLeftWS = SafeNormalize(cross(vecData.cameraForwardWS, _CharacterParams9.xyz));
    
    float3 shiftedLightDirOffset;
    shiftedLightDirOffset.x = vecData.cameraForwardWS.x;
    shiftedLightDirOffset.y = lerp(0.5f, vecData.lightDirWS.y, shadowData.scene);
    shiftedLightDirOffset.z = vecData.cameraForwardWS.z;
    float3 lightDirFinal = vecData.lightDirWS * shadowData.scene + shiftedLightDirOffset * 2.0f;//果然是使用了跟随相机前向的跟踪高光方向
    float3 halfDirWS = SafeNormalize(lightDirFinal) + vecData.viewDirWS;
    vecData.halfDirWS = SafeNormalize(halfDirWS);
    
    vecData.reflectDirWS = reflect(-vecData.viewDirWS, surface.normalWS_withRain);
}

void IntializeEndFieldSurface(Varyings i, uint facing,
    inout EndFieldSurface surface)
{
    surface.depth = i.positionHCS.z;
    surface.baseUV = i.uv.xy;
    surface.baseUV1 = i.uv.zw;
    surface.positionOS = i.positionOS;
    surface.positionWS = i.positionWS;
    surface.positionHCS = i.positionHCS;
    surface.screenUV = GetNormalizedScreenSpaceUV(i.positionHCS.xy);
    surface.screenPos = (uint2)i.positionHCS.xy;
    
    float4 mainTex = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, surface.baseUV);
    float4 pbrMask = SAMPLE_TEXTURE2D(_PBRMask, sampler_PBRMask, surface.baseUV);
    float3 bumpTS = UnpackNormalScale(SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, surface.baseUV), _NormalScale);
    float3 emissionTex = SAMPLE_TEXTURE2D(_EmissionMap, sampler_EmissionMap, surface.baseUV).xyz;

    //params part
    surface.basecolor = mainTex.rgb * _BaseColor.rgb;
    surface.alpha     = mainTex.a * _BaseColor.a;
    surface.basecolor_shadow = surface.basecolor * _ShadowColorBrightness;
    surface.basecolor_shadow = lerp(Luminance(surface.basecolor_shadow), surface.basecolor_shadow, _ShadowColorSaturation);
    surface.metallic = lerp(0, _Metallic, pbrMask.r);
    surface.specularLevel = pbrMask.g;
    surface.anisotropy = 0;
    surface.perceptualsmoothness = pbrMask.a;
    surface.perceptualRoughness = lerp(0, _Roughness, 1 - surface.perceptualsmoothness);
    surface.perceptualRoughness_rain = 0.01f;
    surface.rainMask = 0.0f;
    surface.occlusion = pbrMask.b;
    surface.emission         = emissionTex * _EmissionCol.xyz * _EmissionIntensity;

    //vector part
    float normalLength = length(i.normalWS.xyz);
    float invNormalLength = 1.0 / max(FLT_MIN, normalLength);

    float tangentWS_w = i.tangentWS.w > 0 ? 1 : -1;
    tangentWS_w *= GetOddNegativeScale();
    
    float faced = facing ? 1 : _BackFaceNormalFlip * 2 - 1;
    
    surface.normalWS_raw         = i.normalWS.xyz * rcp(normalLength) * faced;
    surface.tangentWS_raw        = i.tangentWS.xyz * invNormalLength;
    surface.bitangentWS_raw      = cross(i.normalWS.xyz, i.tangentWS.xyz) * tangentWS_w * invNormalLength;
    surface.tangent_W            = i.tangentWS.w;
    
    surface.TBNWS                = float3x3(surface.tangentWS_raw,
                                            surface.bitangentWS_raw,
                                            i.normalWS.xyz * invNormalLength);
    surface.normalOS             = i.normalOS;
    surface.normalWS             = normalize(TransformTangentToWorld(bumpTS, surface.TBNWS)) * faced;
    surface.tangentWS            = 0;
    surface.bitangentWS          = 0;

    surface.normalWS_withRain    = surface.normalWS;
}

void InitializeEndFieldDotData(EndFieldVecData vecData, EndFieldSurface surface, 
    inout EndFieldDotData dotData)
{
    dotData.NdotL = dot(surface.normalWS, vecData.lightDirWS);  //这里没有用withRain的可能是简化细碎的雨水细节
    dotData.NdotL_XZ = dot(surface.normalWS.xz, vecData.lightDirWS.xz);  //这里没有用withRain的可能是简化细碎的雨水细节
    dotData.NdotH = dot(surface.normalWS_withRain, vecData.halfDirWS);
    dotData.NdotV = dot(surface.normalWS_withRain, vecData.viewDirWS);
    
    dotData.VdotH = 0;
    dotData.LdotH = 0;
    
    dotData.NdotL_clamped = saturate(dotData.NdotL);
    dotData.NdotV_clamped = saturate(dotData.NdotV);
    dotData.NdotV2_clamped = dotData.NdotV_clamped * dotData.NdotV_clamped;

    dotData.NgtLdotForwardWS_clamped = saturate(-dot(vecData.lightDirWS_XZ.xz, vecData.cameraForwardWS_XZ.xz));
}

void SurfaceConvertToBSDF(EndFieldVecData vecData, EndFieldDotData dotData, EndFieldSurface surface, inout EndFieldBSDF bsdf)
{
    bsdf.diffuse = ComputeDiffuseColor(surface.basecolor, surface.metallic) * 0.96f;  //0.96f is EndField Feature
    //阴影
    bsdf.diffuse_shadow = ComputeDiffuseColor(surface.basecolor_shadow, surface.metallic) * 0.96f;  //0.96f is EndField Feature
    //更深的阴影
    bsdf.diffuse_shadow2 = bsdf.diffuse_shadow * _CharacterParams0.z * 0.65f;  //0.65f is EndField Feature
    bsdf.diffuse_shadow2 = lerp(Luminance(bsdf.diffuse_shadow2), bsdf.diffuse_shadow2, 1.2f);
    
    bsdf.specularLevel = surface.specularLevel;
    bsdf.F0 = ComputeFresnel0(surface.basecolor, surface.metallic, surface.specularLevel * DEFAULT_SPECULAR_VALUE);
    bsdf.F90 = 1;
    bsdf.perceptualRoughness = surface.perceptualRoughness;
    bsdf.roughness = max(0.0078125, PerceptualRoughnessToRoughness(bsdf.perceptualRoughness));
    bsdf.anisotropy = surface.anisotropy;
    ConvertAnisotropyToRoughness(bsdf.perceptualRoughness, bsdf.anisotropy, bsdf.roughnessT, bsdf.roughnessB);
    bsdf.roughness2 = bsdf.roughnessT * bsdf.roughnessT;
    bsdf.occlusion = surface.occlusion;
    
    bsdf.ramp_NdotL = SampleRampColor_NdotL(vecData, dotData);
    bsdf.ramp_NdotV = SampleRampColor_NdotV(vecData, surface);
}

//region 环境镜面反射
float3 EnvBRDF(EndFieldDotData dotData, EndFieldSurface surface, EndFieldBSDF bsdf)
{
    // 这部分代码实现了环境光照的双向反射分布函数(BRDF)的积分近似。
    // 它使用了高精度的有理函数拟合(Rational Fit)来替代常用的 LUT 采样。
    // 同时包含了一个多重散射(Multi-Scatter)的修正项，用于保持高粗糙度下的能量守恒

    // 预计算粗糙度和 NdotV 的幂次，用于后续的多项式计算
    float envBRDF_roughness2 = surface.perceptualRoughness * surface.perceptualRoughness;
    float rain_NdotV3 = dotData.NdotV2_clamped * dotData.NdotV_clamped;                   //r14.z,     r14.x: rain_NdotV2, r22.x: rain_NdotV
    float envBRDF_roughness6 = envBRDF_roughness2 * envBRDF_roughness2 * envBRDF_roughness2;

    float const1 = 1.0f;

    // 2. 计算 DFG Scale (A) - 对应 F0 的缩放系数
    // 公式形式为有理函数: P(x, y) / Q(x, y)
    // 系数看起来是特定渲染管线（如米哈游 HGRP 或定制管线）预计算的拟合参数
    float termA_Num_X = dot(float2(3.327069997f, const1), float2(dotData.NdotV_clamped, 0.0365463f));
    float termA_Num_Y = dot(float2(-9.047559738f, const1), float2(dotData.NdotV_clamped, 9.0632f));
    float termA_Num   = dot(float2(termA_Num_X, termA_Num_Y), float2(const1, envBRDF_roughness2));

    float termA_Den_X = dot(float3(3.596849918f, -1.367720007f, const1), float3(dotData.NdotV2_clamped, rain_NdotV3, const1));
    float termA_Den_Y = dot(float3(-16.317399978f, const1, 9.229490280f), float3(dotData.NdotV2_clamped, 9.04401f, rain_NdotV3)); 

    float termA_Den_Z = dot(float3(1.0f, 19.788600921f, -20.212299346f), float3(5.56589f, dotData.NdotV2_clamped, rain_NdotV3));
    float termA_Den   = dot(float3(termA_Den_X, termA_Den_Y, termA_Den_Z), float3(const1, envBRDF_roughness2, envBRDF_roughness6));

    float dfgScale = termA_Num / termA_Den; // 变量 A

    // 3. 计算 DFG Bias (B) - 对应 F0 的偏移系数
    float termB_Num_X = dot(float2(-1.285140037f, const1), float2(dotData.NdotV_clamped, 0.99044f));
    float termB_Num_Y = dot(float2(const1, -0.755906999f), float2(1.29678f, dotData.NdotV_clamped));
    float termB_Num   = dot(float2(termB_Num_X, termB_Num_Y), float2(const1, envBRDF_roughness2));

    float termB_Den_X = dot(float3(2.923379898f, 59.418800354f, const1), float3(dotData.NdotV_clamped, rain_NdotV3, const1));
    float termB_Den_Y = dot(float3(const1, -27.030199050f, 222.591995239f), float3(20.3225f, dotData.NdotV_clamped, rain_NdotV3));
    float termB_Den_Z = dot(float3(626.130004882f, 316.627014160f, const1), float3(dotData.NdotV_clamped, rain_NdotV3, 121.563f));
    float termB_Den   = dot(float3(termB_Den_X, termB_Den_Y, termB_Den_Z), float3(const1, envBRDF_roughness2, envBRDF_roughness6));

    float dfgBias = termB_Num / termB_Den; // 变量 B

    // 4. 计算单次散射高光 (Single Scattering Specular)
    // Standard Split-Sum approximation: Specular = F0 * Scale + Bias
    float3 singleScatterSpecular = bsdf.F0 * dfgScale + dfgBias; // 对应 color_266_123

    // 5. 多重散射能量补偿 (Multi-Scattering Energy Compensation)
    // 在高粗糙度下，单次散射会丢失能量导致边缘变暗。这里进行经验性的补偿。
    // integratedEnergy = Scale + Bias 近似代表了对于白色 F0 的总反射能量积分 (Integrated BRDF for F0=1)
    float integratedEnergy = dfgScale + dfgBias; 

    // 计算缺失的能量比例，并归一化
    // (1 - E) / E 是一种常见的用于推导多重散射因子的形式
    // 原始逻辑: multiScatterFactor = (1.0f - integratedEnergy) / integratedEnergy;
    float multiScatterFactor = (1.0f - integratedEnergy) / integratedEnergy;

    // 最终组合：
    // EnvBRDF = SingleScatter + MultiScatter
    // MultiScatter term = F0 * multiScatterFactor * SingleScatter
    // 这种组合方式类似于 Kulla-Conty 近似的简化版：Fms = Eavg * Ems(r) ... 这里的实现看起来是：
    // Result = SingleScatter * (1 + F0 * ((1-E)/E))
    //        = SingleScatter + SingleScatter * F0 * (1/E - 1)
    //        = E_ss + E_ss * F0 * (1/E - 1)  <-- 这是一种定制的补偿混合
    float3 envBRDF = bsdf.F0 * multiScatterFactor * singleScatterSpecular + singleScatterSpecular;
    return envBRDF;
}

real PerceptualRoughnessToMipmapLevel_EndField(real perceptualRoughness, uint maxMipLevel)
{
    float mip = max(0.001, perceptualRoughness);
    mip = log2(mip);
    mip = 1 - mip * 1.2;
    mip = maxMipLevel - mip;
    return mip;
}

float3 EnvSpecular(EndFieldVecData vecData, EndFieldSurface surface)
{
    float mip = PerceptualRoughnessToMipmapLevel_EndField(surface.perceptualRoughness, 6/*EndField is 6*/);
    float3 envSpecular = SAMPLE_TEXTURECUBE_LOD(_IndirSpecCubemap, sampler_LinearRepeat, vecData.reflectDirWS, mip).xyz;
    return envSpecular;
}

//region skin shadowColorLut
#define LUT_A 0.03125f  // 1.0/32
#define LUT_B 0.0302734375f  // 31/1024
#define LUT_C 0.00048828125f  // 0.5/1024
#define LUT_D 0.96875f  // 31/32
#define LUT_E 0.015625f  // 0.5/32

//endregion

// 雨水效果核心参数（可根据需求调整）
#define RAIN_NOISE_SCALE_1 30.0f      // 噪声频率1（控制雨滴密度）
#define RAIN_NOISE_SCALE_2 45.345600128173828125f   // 噪声频率2（补充细节）
#define RAIN_TIME_SPEED_1 3.0f        // 雨滴下落速度1
#define RAIN_TIME_SPEED_2 4.345600128173828125f     // 雨滴下落速度2
#define RAIN_DROP_SIZE_MIN 0.6f       // 雨滴最小尺寸系数
#define RAIN_DROP_SIZE_MAX 1.0f       // 雨滴最大尺寸系数
#define RAIN_ROUGHNESS_WET 0.1f       // 湿润区域粗糙度（越小越光滑）
#define RAIN_METALLIC_WET 1.0f        // 湿润区域金属度（越大反射越强）

//region rain feature func

// 生成伪随机数（输入2D坐标，输出0~1随机值）
float2 GenerateRandom2D(float2 gridPos)
{
    // 随机数种子（固定值，保证随机性）
    float2 randSeed = float2(123.339996337890625f, 456.209991455078125f);
    float2 gridRand = frac(gridPos * randSeed);
    // 点积强化随机性
    gridRand += dot(gridRand, gridRand + 34.345001220703125f).xx;
    return frac(float2(gridRand.x * gridRand.y, gridRand.x + gridRand.y));
}

// 计算单组雨滴强度（输入：坐标、时间系数、尺寸系数，输出：雨滴强度+偏移）
float2 CalculateRainDropIntensity(float2 coord, float timeFactor, out float2 dropOffset)
{
    // 1. 划分噪声格子
    float2 gridPos = floor(coord);
    float2 gridUV = frac(coord) - 0.5f;

    // 2. 生成格子内随机值
    float2 randVal1 = GenerateRandom2D(gridPos);
    float2 randVal2 = GenerateRandom2D(gridPos + 114.51399993896484375f); // 偏移格子避免重复

    // 3. 雨滴中心随机偏移
    dropOffset = ((randVal2 * 2.0f - 1.0f) * 0.25f) + (coord - gridPos) - 0.5f;
    // 雨滴形状非对称拉伸（模拟下落轨迹）
    dropOffset.x *= 1.25f;
    dropOffset.y *= dropOffset.y < 0.0f ? 1.25f : 0.75f;

    // 4. 雨滴范围判断（距离中心越近强度越高）
    float dropLength = length(dropOffset);
    float dropSize = 0.25f * lerp(0.60f, 1.0f, randVal1.x);
    float rangeIntensity = smoothstep(dropSize, 0.0f, dropLength);

    // 5. 时间窗口判断（模拟雨滴下落的生命周期）
    float timeCycle = frac(timeFactor + randVal1.x);
    float2 timeIntensity;
    timeIntensity.x = smoothstep(0.2f, 0.22f, timeCycle) * smoothstep(0.5f, 0.2f, timeCycle);


    // 6. 最终雨滴强度（范围+时间双重过滤）
    timeIntensity.x *= step(0.001f, rangeIntensity);
    timeIntensity.y = randVal2.y;

    //输出
    dropOffset = clamp(dropOffset / dropSize, -1.0f, 1.0f) * timeIntensity.x * lerp(0.5f, 1.0f, randVal2.x);
    // return float2(dropLength, 0);
    return timeIntensity;
}

//cbt-3 skin
void ApplyRainFeature(VecData vecData, float4 _WaterParam, float perceptualRoughness, float metallicRatio,
    inout float rainIntensity, inout float wetPerceptualRoughness, inout float wetMetallicRatio, inout float3 wetNormalWS)
{
    // float4 _WaterParam = _20_m0_12_m7;
    float _PosScale = _cb0_170.z;
    // float baseMetallicRatio = _46_m1;

    float depthDiff = _WaterParam.z - vecData.positionWS.y;
    float posRainIntensity = smoothstep(-0.2f, 0.15f, depthDiff) * _WaterParam.y;
    float enableRain = _WaterParam.x + posRainIntensity;

    if (9.99999975e-005 < enableRain)
    {
        // 最终雨水强度权重（取全局和位置的最大值）
        float finalRainWeight = max(_WaterParam.x, posRainIntensity);

        bool switchDir = _RainFilpObjectDir > 0.01f;

        float3 posOS = vecData.positionOS;
        posOS = switchDir ? posOS.xzy * float3(1.0f, 1.0f, -1.0f) : posOS.xyz;
        float3 scaledPosOS = posOS * _PosScale;

        float3 normalAbs = abs(vecData.normalWS_raw);
        normalAbs = switchDir ? normalAbs.xzy : normalAbs.xyz;
        float3 normalWeight = max(pow(normalAbs - 0.2f, 10.0f), 0.0f);
        // 归一化权重，确保三个轴权重和为1
        float3 normalWeightNLZ = normalWeight / dot(normalWeight, float3(1.0f, 1.0f, 1.0f));

        // 时间因子（控制雨滴下落动画）
        float timeFactor1 = _Time.x * RAIN_TIME_SPEED_1;
        float timeFactor2 = _Time.x * RAIN_TIME_SPEED_2;

        // 存储三组雨滴的强度和偏移
        float2 dropIntensity[6];
        float2 dropOffset[6];

        // 第一组：XZ轴（频率1）
        dropIntensity[0] = CalculateRainDropIntensity(scaledPosOS.xz * RAIN_NOISE_SCALE_1, timeFactor1, dropOffset[0]);
        // 第二组：XY轴（频率1）
        dropIntensity[1] = CalculateRainDropIntensity(scaledPosOS.xy * RAIN_NOISE_SCALE_1, timeFactor1, dropOffset[1]);
        // 第三组：ZY轴（频率1）
        dropIntensity[2] = CalculateRainDropIntensity(scaledPosOS.zy * RAIN_NOISE_SCALE_1, timeFactor1, dropOffset[2]);
        // 第四~六组：补充频率2的细节
        dropIntensity[3] = CalculateRainDropIntensity(scaledPosOS.xz * RAIN_NOISE_SCALE_2, timeFactor2, dropOffset[3]);
        dropIntensity[4] = CalculateRainDropIntensity(scaledPosOS.xy * RAIN_NOISE_SCALE_2, timeFactor2, dropOffset[4]);
        dropIntensity[5] = CalculateRainDropIntensity(scaledPosOS.zy * RAIN_NOISE_SCALE_2, timeFactor2, dropOffset[5]);
        // 计算雨水流动的切线方向（与世界Y轴垂直）
        float3 tangentDir = cross(vecData.normalWS, float3(0.0f, 1.0f, 0.0f));
        float3 flowDir = length(tangentDir) > 1e-4f ? normalize(tangentDir) : float3(-1.0f, 0.0f, 0.0f);

        // 合并三组雨滴的偏移，按法线权重加权
        //xy
        float2 part1DropOffset = dropOffset[0] * normalWeightNLZ.y + dropOffset[1] * normalWeightNLZ.z + dropOffset[2] * normalWeightNLZ.x;
        float2 part2DropOffset = dropOffset[3] * normalWeightNLZ.y + dropOffset[4] * normalWeightNLZ.z + dropOffset[5] * normalWeightNLZ.x;
        //zw
        float2 part1Intensity = dropIntensity[0] * normalWeightNLZ.y + dropIntensity[1] * normalWeightNLZ.z + dropIntensity[2] * normalWeightNLZ.x;
        float2 part2Intensity = dropIntensity[3] * normalWeightNLZ.y + dropIntensity[4] * normalWeightNLZ.z + dropIntensity[5] * normalWeightNLZ.x;

        float2 totalIntensity = max(part1Intensity, part2Intensity);
        float2 totalDropOffset = part1DropOffset + part2DropOffset;
        // 最终雨水强度（结合全局开关）
        rainIntensity = totalIntensity.x * step(1.0f - _WaterParam.x, totalIntensity.y - 0.1f);
        bool2 cmp_Intensity = rainIntensity > 0.001f;
        float2 rainDropOffset = float2(cmp_Intensity.x ? totalDropOffset.x : 0.0f, cmp_Intensity.y ? totalDropOffset.y : 0.0f);

        // 法线扰动：原始法线 插值到 流动方向法线
        float3 perturbedNormal = lerp(vecData.normalWS, flowDir, rainDropOffset.x);
        perturbedNormal = normalize(lerp(perturbedNormal, cross(vecData.normalWS, flowDir), rainDropOffset.y));
        wetNormalWS = lerp(vecData.normalWS, perturbedNormal, rainIntensity);

        // 粗糙度插值：基础粗糙度 -> 半湿润 -> 完全湿润
        float roughHalfWet = lerp(perceptualRoughness, 0.5f, finalRainWeight);
        wetPerceptualRoughness = lerp(roughHalfWet, RAIN_ROUGHNESS_WET, rainIntensity);

        // 金属度插值：基础金属度 -> 半湿润 -> 完全湿润
        float metallicHalfWet = lerp(metallicRatio, RAIN_METALLIC_WET, finalRainWeight);
        wetMetallicRatio = lerp(metallicHalfWet, RAIN_METALLIC_WET, rainIntensity);
    }
    else
    {
        rainIntensity = 0;
        perceptualRoughness = perceptualRoughness + 0.0f;
        metallicRatio = metallicRatio + 0.0f;
        wetNormalWS = vecData.normalWS;
    }
}
//endregion

void InitialVecData(Varyings i, uint facing, float3 bumpTS, inout VecData vecData)
{
    vecData.positionHCS = i.positionHCS;
    vecData.positionWS = i.positionWS;
    vecData.normalWS_raw = SafeNormalize(i.normalWS.xyz);
    vecData.bitangentWS_raw = SafeNormalize(i.bitangentWS.zxy);
    vecData.normalWS_raw = facing > 0 ? vecData.normalWS_raw : -vecData.normalWS_raw;
    float3x3 TBN = float3x3(i.tangentWS.xyz, i.bitangentWS.xyz, vecData.normalWS_raw.xyz);
                
    vecData.normalWS = SafeNormalize(TransformTangentToWorld(bumpTS, TBN));
    vecData.bitangentWS = normalize(vecData.bitangentWS_raw.zxy);
    vecData.tangentWS = cross(vecData.bitangentWS, vecData.normalWS);
    vecData.bitangentWS = cross(vecData.normalWS, vecData.tangentWS);

    vecData.viewDirWS = SafeNormalize(i.viewDirWS.xyz);
    vecData.reflectDirWS = reflect(vecData.viewDirWS, vecData.normalWS);
}

void InitialDotData(VecData vecData, inout DotData dotData)
{
    float ndotl = dot(vecData.normalWS, vecData.lightDirWS);
    dotData.ndl = max(0, ndotl);
    float ndoth = dot(vecData.normalWS, vecData.halfDirWS);
    dotData.ndh = max(0, ndoth);
    float vdoth = dot(vecData.viewDirWS, vecData.halfDirWS);
    dotData.vdh = max(0, vdoth);
    #ifdef _SHADOW_RAMP
    float ldoth = dot(vecData.lightDirWS, vecData.halfDirWS);    //dot(lightDirWS, halfDir)
    dotData.ldh = max(0, ldoth);
    #endif
}

void UnpackHairNormalMap(inout real3 normal1, inout real3 normal2,
    real4 packedNormal, real scale1 = 1.0, real scale2 = 1.0
    )
{
    real3 nor1, nor2;
    nor1.xy = packedNormal.xy * 2.0 - 1.0;
    nor1.z = max(1.0e-16, sqrt(1.0 - saturate(dot(nor1.xy, nor1.xy))));
    nor1.xy *= scale1;
    normal1 = nor1;
    
    nor2.xy = packedNormal.zw * 2.0 - 1.0;
    nor2.z = max(1.0e-16, sqrt(1.0 - saturate(dot(nor2.xy, nor2.xy))));
    nor2.xy *= scale2;
    normal2 = nor2;
}

//region anisotropy specular
float AnisotropicSpecular(float3 T, float3 H)
{
    float dotTH = dot(T, H);
    float sinTH = sqrt(1.0f - dotTH * dotTH);
    return max(sinTH, 9.9999997473787516355514526367188e-05f);;
}

//region fog feature
static const float K_LOG2E = 1.44269502162933349609375f; // 1/ln(2)
static const float K_PI = 3.14159265359f;

// 辅助函数：计算高度雾的指数积分
// 对应: (1 - exp(-falloff * dist)) / falloff * exp(-falloff * (camY - base))
// 参数说明：
// vectorY:   视线向量的 Y 分量 (posWS.y - camPos.y)
// camHeight: 相机世界高度
// baseHeight: 雾起始高度 (_ExponentialFogParams.x 或 .z)
// falloff:    高度衰减系数 (_ExponentialFogParams.z 或 .x)
// density:    雾密度 (_ExponentialFogParams.y)
float ComputeHeightFogIntegral(float vectorY, float camHeight, float baseHeight, float falloff, float density)
{
    // --- 1. 计算相机处的指数项 (Camera Term) ---
    // Text 2: Line 196-200
    // exponent = (CamY - Base) * Falloff
    float camExponent = (camHeight - baseHeight) * falloff;
    camExponent = max(camExponent, -127.0f); // 保护 exp2
    
    // camTerm = Density * 2^(-exponent)
    float camTerm = density * exp2(-camExponent);

    // --- 2. 计算积分变量 x (Delta Height Term) ---
    // Text 2: Line 201
    // x = VectorY * Falloff
    float x = vectorY * falloff;
    
    // --- 3. 计算准确解 (Exact Solution) ---
    // Text 2: Line 202-205
    // result = (1 - 2^(-x)) / x
    // 这里 text 2 也对 x 做了 max(-127) 保护，虽然主要影响 pow 的结果
    float x_safe = max(x, -127.0f); 
    float expVal = exp2(-x_safe);
    float resExact = (1.0f - expVal) / x_safe; // 如果 x=0 这里会 NaN，但后面会处理

    // --- 4. 计算近似解 (Taylor Approximation) ---
    // Text 2: Line 212
    // 用于 x 接近 0 时的情况
    // Taylor: ln(2) - x * (ln(2)^2 / 2)
    // 0.6931472 = ln(2)
    // 0.2402265 = 0.480453 / 2 = (ln(2)^2) / 2
    float resApprox = 0.6931472f - x_safe * 0.2402265f;

    // --- 5. 选择分支 (Branchless Selection) ---
    // Text 2: Line 213-214
    // 阈值 5.96e-08f (即 0.00000006)
    float finalInt = (abs(x_safe) > 5.96e-08f) ? resExact : resApprox;

    // --- 6. 组合结果 ---
    // Text 2: Line 215
    return camTerm * finalInt;
}

// =================================================================================
// 主雾效逻辑
// =================================================================================
float3 ApplyFog(float3 sceneColor, float3 posWS, float3 screenPos)
{
    return 0;
}

#endif