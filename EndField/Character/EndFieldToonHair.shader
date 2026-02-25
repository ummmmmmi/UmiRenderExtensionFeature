Shader "DanbaidongRP/EndFieldToon/CBT-2/Hair"
{
    Properties
    {
        // Only Debug
        [FoldoutBegin(_FoldoutDebug_End)]_FoldoutDebug("Debug", Float) = 0
            [Toggle]
            _ShowAlbedo                              ("ShowAlbedo", Float)                   = 0
        [FoldoutEnd]_FoldoutDebug_End("_FoldoutDebug_End", Float) = 0
        
        // GPU Animation
        [FoldoutBegin(_FoldoutGpuAnimationEnd, _USE_GPU_ANIMATION)]_FoldoutGpuAnimation("GPU Animation", Float) = 0
        [HideInInspector]_USE_GPU_ANIMATION("_USE_GPU_ANIMATION", Float) = 0
            _GPU_Animation_Tint                      ("GPU动画融合度", Range(0, 1))            = 1
        [FoldoutEnd]_FoldoutGpuAnimationEnd("_FoldoutGpuAnimationEnd", Float) = 0
        
        // GPU Animation Structured
        [FoldoutBegin(_FoldoutGpuAnimation_Structured_End, _USE_GPU_ANIMATION_STRUCTURED)]_FoldoutGpuAnimation_Structured("GPU Animation Structured", Float) = 0
        [HideInInspector]_USE_GPU_ANIMATION_STRUCTURED("_USE_GPU_ANIMATION_STRUCTURED", Float) = 0
//            _GPU_Animation_Structured_Tint           ("GPU动画融合度", Range(0, 1))            = 1
            _BoneMatricesCount                       ("GPU动画buffer数量", Float)              = 0
            _BoneIndexOffset                         ("BoneIndexOffset", Vector)              = (184905, 183633, 514, 1)
            _BoneMaxCount                            ("BoneMaxCount", Range(0, 4))            = 4
        [FoldoutEnd]_FoldoutGpuAnimation_Structured_End("_FoldoutGpuAnimation_Structured_End", Float) = 0
        
        // frag data
        [FoldoutBegin(_FoldoutFragData_End)]_FoldoutFragData("Frag Data", Float) = 0
            _FragDataCount                           ("FragDatabuffer数量", Float)           = 0
        [FoldoutEnd]_FoldoutFragData_End("_FoldoutFragData_End", Float) = 0
        
        [FoldoutBegin(_FoldoutTexEnd)]_FoldoutTex("Textures", Float) = 0
            _BaseColor                              ("BaseColor", Color)                    = (1,1,1,1)
            _BaseMap                                ("BaseMap(diff alpha)", 2D)             = "white" {}
            [NoScaleOffset]_PBRMask                 ("PBRMask(使用程序化切线mask 高光强度 ao range2mask)", 2D)  = "white" {}
     
            [NoScaleOffset]_NormalMap               ("NormalMap", 2D)                       = "bump" {}
            _NormalScale                            ("NormalScale", Range(0, 2))            = 1
        
            [NoScaleOffset]_SplitNormalMap          ("'Hair Normal Map' {}", 2D)            = "gray" {}
            _SpecBumpScale                          ("'Spec Scale' {}", Range(0, 2))        = 1
            
            
        [FoldoutEnd]_FoldoutTexEnd("_FoldoutEnd", Float) = 0
        
        [FoldoutBegin(_DecodeEnd)]_Decode("_Decode", Float) = 0
            [Enum(Opaque, 0, Transparent, 1)]
            _SurfaceType                            ("'Surface Type' {Dropdown:{Opaque:{_BlendMode, _OutlineTransparent, _TransparentDepthWrite}, Transparent:{}}}", Float) = 0
            [Enum(On, 0, Off, 1)]
            _BackFaceNormalFlip                     ("'Back Face Normal Flip' {Dropdown:{On:{}, Off:{}}}", Float) = 0
            _ShadowColorBrightness                  ("'Shadow Color Brightness' {}", Range(0, 1))       = 0.5
            _ShadowColorSaturation                  ("'Shadow Color Saturation' {}", Range(0, 2))       = 1
            [Space(10)]
            [ToggleUI] _SpecRampIridescentMode      ("'彩虹色模式(镭射塑料请勾选)' {}", Float)              = 0
        
            [Space(10)]
            [HideInInspector] _ExposureParams                         ("ExposureParams.x", Vector)                          = (1.0, 0.0, 0.0, 0.0)
            [HideInInspector] _CharacterParams0                       ("CharacterParams0 z: shadow draken", Vector)         = (1.0, 1.05, 0.55, 0.8)
            [HideInInspector] _CharacterParams1                       ("CharacterParams1 y: enable indirectSHMap w: use custom light dir", Vector) = (0.0, 1.0, 1.0, 1.0)
            [HideInInspector] _CharacterParams2                       ("CharacterParams2 xyz: default indirectDiffuseColor w: indirectDiffuse Intensity", Vector) = (0.8783069, 0.9302293, 1.1216931, 0.3)
            [HideInInspector] _CharacterParams3                       ("CharacterParams3 w: intensity", Vector)             = (1.18701, 0.9272287, 0.8129899, 1.0)
            [HideInInspector] _CharacterParams4                       ("CharacterParams4 xyz:custom light dir, a: intensity", Vector)  = (0.1972383, 0.5299193, 0.8247925, 0.0)
            [HideInInspector] _CharacterParams5                       ("CharacterParams5 xyz:dirLightColor, a: intensity", Vector)  = (1.0, 1.0, 1.0, 1.0)
            [HideInInspector] _CharacterParams6                       ("CharacterParams6 xyz:dir", Vector)                  = (0.0, 0.0, 1.0, 0.0)
            [HideInInspector] _CharacterParams7                       ("CharacterParams7 x: y: z: ", Vector)                = (0.15, 1.5, 0.5, 0.4)

            /*[HideInInspector]*/ _EnvironmentGlobalParams0               ("EnvironmentGlobalParams0", Vector)                  = (0.0, 0.0, 0.0, 0.0)
            /*[HideInInspector]*/ _IVParam0                               ("IVParam0", Vector)                                  = (0.0, 0.0, 0.0, 0.0)
            /*[HideInInspector]*/ _IVParam1                               ("IVParam1", Vector)                                  = (0.0, 0.0, 0.0, 0.0)
            [Space]
            /*[HideInInspector]*/ _IVDefaultSHAr                          ("_IVDefaultSHAr", Vector)                            = (0.0, 0.0, 0.0, 0.0)
            /*[HideInInspector]*/ _IVDefaultSHAg                          ("_IVDefaultSHAg", Vector)                            = (0.0, 0.0, 0.0, 0.0)
            /*[HideInInspector]*/ _IVDefaultSHAb                          ("_IVDefaultSHAb", Vector)                            = (0.0, 0.0, 0.0, 0.0)
        
        [FoldoutEnd]_DecodeEnd("_DecodeEnd", Float) = 0
        
        [FoldoutBegin(_RimEnd)]_Rim("Rim", Float) = 0
            _RimLightPosWS                          ("cb1_3 xyz: Rim lightPos", Vector)                                    = (0.0, 0.0, 0.0, 1.0)
            _CharacterParams8                       ("CharacterParams8 xyz: rimFinalColor w: RimIntensity", Vector)        = (0.0, 0.0, 0.0, 1.0)
            _CharacterParams9                       ("CharacterParams9 xyz: Rim dir", Vector)                              = (0.0, -1.0, 0.0, 1.0)
        [FoldoutEnd]_RimEnd("_RimEnd", Float) = 0
        
        [FoldoutBegin(_RainingEnd)]_Raining("Raining", Float) = 0
            _CharacterRainEffectTex                 ("PointRainMap(rg:rain normal, b:mask, a:flow dir)", 2D)             = "gray" {}
            _CharacterRainStreakTex                 ("Vertical RainMap(rg:rain normal, b:mask, a:flow dir)", 2D)         = "gray" {}
            /*[HideInInspector]*/ [Toggle]_RainFilpObjectDir              ("切换方向", Float)                              = 0
            /*[HideInInspector]*/ _CharacterParams10                      ("CharacterParams10 x: use rain custom ppt, y: up rain intensity, z: pos scale, w: wet height", Vector)                = (1.0, 1.0, 2.0, 0.0)
            [HideInInspector] _RainEffectIntensity                    ("Rain Effect Intensity", Range(0, 1))          = 0
            [HideInInspector] _WetEffectWorldSpaceHeight              ("Wet Effect World Space Height", Float)        = -1000
            [HideInInspector] _WetEffectIntensity                     ("Wet Effect Intensity", Range(0, 1))           = 0
        [FoldoutEnd]_RainingEnd("_RainingEnd", Float) = 0
        
        [FoldoutBegin(_FoldoutPBRPropEnd)]_FoldoutPBRProp("PBR Properties", Float) = 0
            _Metallic                               ("Metallic", Range(0, 1))               = 0.0
            _Roughness                              ("Roughness", Range(0, 1))              = 0.5
            _Occlusion                              ("Occlusion", Range(0, 1))              = 1
        [FoldoutEnd]_FoldoutPBRPropEnd("_FoldoutPBRPropEnd", Float) = 0
        
        //Anisotropy
        [FoldoutBegin(_FoldoutAnisotropyEnd)]_FoldoutAnisotropy("Anisotropy Properties", Float) = 0
            _AnisotropyValue                        ("'Anisotropy Offset1' {}", Range(0, 1))          = 0.35
            _AnisotropyDirX                         ("'Anisotropy Direction X' {}", Range(-1, 1))   = 0
            _AnisotropyIntensity                    ("'Anisotropy Intensity' {}", Range(0, 3))      = 1
            _AnisotropyEdgeFade                     ("'Anisotropy Edge Fade' {}", Range(0.01, 10))  = 1
            _AnisotropyValue2                       ("'Anisotropy Offset2' {}", Range(0, 1))         = 0.4
            _AnisotropyRange2                       ("'Anisotropy Range2' {}", Range(-1, 1))        = 0
            _AnisotropyColor2                       ("'Anisotropy Color2' {}", Color)               = (0, 0, 0, 1)
        [FoldoutEnd]_FoldoutAnisotropyEnd("_FoldoutAnisotropyEnd", Float) = 0
        
        // SpecularLine
        [FoldoutBegin(_FoldoutSpecularLineEnd, _SPECULAR_LINE)]_FoldoutSpecularLine("SpecularLine Properties", Float) = 0
        [HideInInspector]_SPECULAR_LINE("_SPECULAR_LINE", Float) = 0
            [Toggle]_UseLineMap                     ("'Use Line Map' {Toggle:{On:{_LineAmount}, Off:{_LineMap}}}", Float) = 0
            _LineMap                                ("'Line Map' {}", 2D)                           = "black" {}
            _LineAmount                             ("'Line Amount' {}", Float)                     = 300
            _LineValue                              ("'Line Value' {}", Range(0, 1))                = 0
            _LineRange                              ("'Line Range' {}", Range(-1, 1))               = 0
            _LineIntensity                          ("'Line Intensity' {}", Range(0, 1))            = 0
            _LineSaturation                         ("'Line Saturation' {}", Range(0, 10))          = 1
        [FoldoutEnd]_FoldoutSpecularLineEnd("_FoldoutSpecularLineEnd", Float) = 0
        
        // Shining Decal
        [FoldoutBegin(_FoldoutShining_DecalEnd, _DECAL_UV)]_FoldoutShining_Decal("Shining Decal Properties", Float) = 0
        [HideInInspector]_DECAL_UV("_DECAL_UV", Float) = 0
            _DecalMap                               ("DecalMap(diff alpha)", 2D)         = "black" {}
            [Title(DecalMask)]
            _DecalAnisoUIntensity                   ("DecalAnisoUIntensity", Range(0, 1))    = 0.05
            _DecalMaskPower                         ("DecalMaskPower", Range(0.01, 30))      = 1
            _DecalMaskMin                           ("DecalMaskMin", Range(0, 2))            = 0
            _DecalMaskMax                           ("DecalMaskMax", Range(0, 2))            = 1
        
            [Title(Fwidth)]
            _FwidthRange                            ("FwidthRange", Range(0, 1))             = 1
        
            [Title(CombineDecalColor)]
            _DecalUVOffset                          ("DecalUVOffset", Range(0, 2))           = 1
            _DecalUVScale                           ("DecalUVScale", Range(0, 50))           = 20
            [Title(Decal Rough Metal)]
            _DecalRoughnessScale                    ("DecalRoughnessScale", Range(0, 2))     = 1
            _DecalMetallicScale                     ("DecalMetallicScale", Range(0, 2))      = 1
        
            [Title(CombineDecalColor)]
            _DecalCenterIntensity                   ("DecalCenterIntensity", Range(0, 50))   = 1
            _DecalRimIntensity                      ("DecalRimIntensity", Range(0, 50))      = 1
        [FoldoutEnd]_FoldoutShining_DecalEnd("_FoldoutShining_DecalEnd", Float) = 0

        // Direct Light
        [FoldoutBegin(_FoldoutDirectLightEnd)]_FoldoutDirectLight("Direct Light", Float) = 0
            [HDR]_SelfLight                         ("SelfLightColor", Color)                = (1,1,1,1)
            _MainLightColorLerp                     ("Unity Light or SelfLight", Range(0, 1))= 0.5
            [HDR]_SelfAddLightColor                 ("SelfAddLightColor", Color)             = (1,1,1,1)
            _AddLightColorLerp                      ("Unity AddLight or SelfAddLight", Range(0, 1))= 0.5
            _DirectOcclusion                        ("DirectOcclusion", Range(0, 1))         = 0.1
             
            [Title(Shadow)]
            [HideInInspector] _DirectionalShadowParams                ("DirectionalShadowParams x: ", Vector)     = (1.0, 2.00, 0.000434, 6400.00)
            [HideInInspector] _DirectionalShadowParams2               ("DirectionalShadowParams2 x: z: ", Vector) = (0.0, 0.00, 1.00, 0.00)
            _ShadowColor                            ("ShadowColor", Color)                   = (0,0,0,1)
            _ShadowOffset                           ("ShadowOffset", Range(-1, 1))           = 0.5
            _ShadowSmoothNdotL                      ("ShadowSmoothNdotL", Range(0, 1))       = 0.25
            _ShadowSmoothScene                      ("ShadowSmoothScene", Range(0, 1))       = 0.1
            _ShadowStrength                         ("ShadowStrength", Range(0, 1))          = 1.0
        [FoldoutEnd]_FoldoutDirectLightEnd("_FoldoutEnd", Float) = 0

        // Ramp
        [FoldoutBegin(_FoldoutShadowRampEnd, _SHADOW_RAMP)]_FoldoutShadowRamp("ShadowRamp", Float) = 0
        [HideInInspector]_SHADOW_RAMP("_SHADOW_RAMP", Float) = 0
            [Ramp]_ShadowRampTex                    ("ShadowRampTex", 2D)                   = "white" { }
            _SpecRampMap                        ("VerticalRampTex", 2D)                 = "white" { }
        [FoldoutEnd]_FoldoutShadowRampEnd("_FoldoutEnd", Float) = 0

        // Indirect Light
        [FoldoutBegin(_FoldoutIndirectLightEnd)]_FoldoutIndirectLight("Indirect Light", Float) = 0
            [Title(directSpecularColor)]
            [Toggle]_EnableLaser                    ("使用镭射", Float)                      = 0
            [NoScaleOffset]_LaserMap                ("LaserMap(color alpha)", 2D)           = "black" {}
            [HDR]_LaserDefaultColor                 ("LaserDefaultColor", Color)            = (0.8783069, 0.9302293, 1.1216931, 1)
            _LaserIntensity                         ("LaserIntensity", Range(0,1))          = 0.3
            [Space]
            _T3                                     ("T3", 3D)                              = "black" {}
            _T4                                     ("T4", 3D)                              = "black" {}
            _T5                                     ("T5", 3D)                              = "black" {}
            [Toggle]
            _SimplerSH                              ("简化SH", Float)                        = 0
            [HDR]_SelfEnvColor                      ("SelfEnvColor", Color)                 = (0.5,0.5,0.5,0.5)
            _EnvColorLerp                           ("Unity SH or SelfEnv", Range(0, 1))    = 0.5
            _IndirDiffUpDirSH                       ("使用朝上方向的法线SH", Range(0, 1))      = 0.0
            _IndirDiffIntensity                     ("IndirDiffIntensity", Range(0, 1))     = 1.0
            [Title(Specular)]
            [Toggle(_INDIR_CUBEMAP)]_INDIR_CUBEMAP("_INDIR_CUBEMAP", Float)         = 0
            [NoScaleOffset]
            _IndirSpecCubemap                       ("SpecCube", Cube)                      = "black" {}

            _IndirSpecCubeWeight                    ("SpecCubeWeight", Range(0, 1))         = 0.5
            _IndirSpecIntensity                     ("IndirSpecIntensity", Range(0.01, 5))  = 1.0
        [FoldoutEnd]_FoldoutIndirectLightEnd("_FoldoutEnd", Float) = 0

        // Emission, Rim, etc.
        [FoldoutBegin(_FoldoutEmissRimEnd)]_FoldoutEmissRim("Emission, Rim, etc.", float) = 0
            [Title(Emission)]
            _EmissionMap                            ("EmissionMap(diff alpha)", 2D)         = "black" {}
            [HDR]_EmissionCol                       ("EmissionCol", Color)                  = (0,0,0,1)
            _EmissionIntensity                      ("_EmissionIntensity", Range(0, 10))    = 1

            [Title(RimLight)]
            [HDR]_DirectRimFrontCol                 ("DirectRimFrontCol", Color)            = (1,1,1,0.5)
            [HDR]_DirectRimBackCol                  ("DirectRimBackCol", Color)             = (0.2,0.2,0.2,0.5)
            _DirectRimWidth                         ("DirectRimWidth", Range(0, 10))        = 2.5
            _PunctualRimWidth                       ("PunctualRimWidth", Range(0, 10))      = 2.75
        [FoldoutEnd]_FoldoutEmissRimEnd("_FoldoutEnd", float) = 0
        
        //VFX Special
        [FoldoutBegin(_FoldoutVFXSpecialEnd, _CHARACTER_VFX_SPECIAL)]_FoldoutVFXSpecial("VFX Special", float) = 0
        [HideInInspector] _EnableCharacterVFX       ("'Character VFX' {Feature:{Color:7}}", Float)    = 0
            [Title(VFX Special)]
            _VFXSpecialMainTex                      ("VFX Special Main Tex(rgb:mask alpha)", 2D)      = "white" {}
            _VFXSpecialBlendTex                     ("VFX Special Blend Tex(?)", 2D)                  = "white" {}
            [HDR]_VFXColor                          ("'VFX Color' {}", Color)               = (1, 1, 1, 1)
            _VFXColorIntensity                      ("'VFX Color Intensity (Default 1)' {}", Range(1, 100)) = 1
            _VFXColorAlpha                          ("'VFX Color Alpha (Default 1)' {}", Range(0, 10)) = 1
            [Enum(UV1, 0, UV2, 1)] _VFXMainUVSet    ("'Main UV Set' {}", Float)             = 0
            [ToggleUI] _UseVFXMainTexAsAlpha        ("'UseMainTexAsAlpha' {}", Float)       = 0
            _VFXSpecialParam                        ("'VFX Special Param(XY: MainTex, ZW: BlendTex)' {}", Vector) = (0, 0, 0, 0)
            [HDR] [Gamma] _VFXBlendTint             ("'Blend Tint' {}", Color)               = (1, 1, 1, 1)
            [HDR] [Gamma] _VFXFresnelColor          ("'Fresnel Color' {}", Color)           = (1, 1, 1, 1)
            _VFXFresnelBias                         ("'Fresnel Bias(Default:0)' {}", Range(-1, 2)) = 0
            _VFXFresnelAffectOpacity                ("'Fresnel Affect Opacity' {}", Range(0, 1)) = 1
            _VFXFresnelPower                        ("'Fresnel Power(Default:1)' {}", Range(1, 100)) = 1
            [ToggleUI] _VFXFresnelFlip              ("'Fresnel Flip' {}", Float)            = 0.0
            _SpecialDissolveScheduleOffset          ("'Dissolve Schedule Offset' {}", Range(0, 2)) = 0
        [FoldoutEnd]_FoldoutVFXSpecialEnd("_FoldoutVFXSpecialEnd", float) = 0
        

        // Outline
        [FoldoutBegin(_FoldoutOutlineEnd, PassSwitch, CharacterOutline)]_FoldoutOutline("Outline", float) = 0
            [Toggle]
            _OutLineNormalSource                    ("Smooth Normal Source UV3, On, UV1, Off", float)         = 0
            _OutlineColor                           ("Outline Color", Color)                = (0, 0, 0, 0.8)
            _OutlineWidth                           ("Width", Range(0, 10))                 = 1.0
            _OutlineZOffset                         ("描边深度偏移", Range(-1, 1))            = 0.0
        
            [Header(Custom_Curve)]
            _Outline_Custom_CurveDistance           ("自定义曲线距离", float)                  = 539.99994
            _Outline_Custom_CurveDistance_Tint      ("自定义曲线距离_强度",  Range(0, 1))     = 0
        
            [Header(DepthFade)]
            [Toggle]_Use_Outline_DepthFade          ("使用深度渐变", float)                    = 0.0
            _OutlineWidth_DepthFade                 ("深度渐变_描边宽度, ", Range(0, 5))        = 1.0
            _Outline_DepthFade_Offset               ("深度渐变_偏移", float)                   = 0.0
            _Outline_DepthFade_Scale                ("深度渐变_缩放范围", float)                = 1.0
        
            [Title(Lighting)]
            [HDR]_OutlineDirectLightingColor        ("DirectColor", Color)                  = (1,1,1,0.5)
            _OutlineDirectLightingOffset            ("DirectOffset", Range(-1, 1))          = -1
            [HDR]_OutlinePunctualLightingColor      ("PunctualColor", Color)                = (1,1,1,0.5)
            _OutlinePunctualLightingOffset          ("PunctualOffset", Range(-1, 1))        = -1
        [FoldoutEnd]_FoldoutOutlineEnd("_FoldoutEnd", float) = 0
        
        [FoldoutBegin(_FogEnd)]_Fog("Fog", Float) = 0
            
            _IntegratedLightScattering              ("IntegratedLightScattering", 3D)                    = "black" {}
            // ====================================== 角色控制参数 ======================================
            [HideInInspector] _CharacterParams11 ("Character Control Param 11 (w:分支阈值<0.5启用雾)", Vector) = (0,0,0,1.0)
            [Title(Base Fog)]
            // ====================================== 大气散射参数 ======================================
            [HideInInspector] _AtmosphereFogParams0 ("Atmosphere Fog Param 0 (xyz:太阳光颜色强度; w:标高H)", Vector)          = (1,1,1,1000)
            [HideInInspector] _AtmosphereFogParams1 ("Atmosphere Fog Param 1 (xyz:Raylengh散射系数; w:米氏各向异性g)", Vector) = (0.005,0.002,0.001,0.8)
            [HideInInspector] _AtmosphereFogParams2 ("Atmosphere Fog Param 2 (xyz:吸收系数; w:基准高度)", Vector)             = (0.001,0.001,0.001,0)
            [HideInInspector] _AtmosphereFogParams3 ("Atmosphere Fog Param 3 (xyz:Mie散射系数; w:路径长度参数)", Vector)       = (0.01,0.008,0.005,0)
            [HideInInspector] _AtmosphereFogParams4 ("Atmosphere Fog Param 4 (xyz:环境光项; w:最大高度)", Vector)             = (0.2,0.2,0.2,10000)
            [HideInInspector] _AtmosphereFogParams5 ("Atmosphere Fog Param 5 (xyz:太阳方向; w:路径长度缩放)", Vector)          = (0,1,0,0.1)
            [Title(Volumetric Fog)]
            // ====================================== 体积雾参数 ======================================
            [HideInInspector] _VolumetricFogParams0 ("Volumetric Fog Param 0 (z:Z切片总数; w:最大Z深度)", Vector)             = (0,0,100,5000)
            [HideInInspector] _VolumetricFogParams1 ("Volumetric Fog Param 1 (x:深度对数缩放; y:深度对数偏移; z:对数Z转切片缩放)", Vector) = (0.001,1,0.1,0)
            [HideInInspector] _VolumetricFogParams2 ("Volumetric Fog Param 2 (x:U缩放; y:V缩放)", Vector)                    = (0.001,0.001,0,0)
            [HideInInspector] _VolumetricFogParams3 ("Volumetric Fog Param 3 (z:最大距离)", Vector)                          = (0,0,10000,0)
            [HideInInspector] _VolumetricFogParams4 ("Volumetric Fog Param 4 (w:抖动强度)", Vector)                          = (0,0,0,1)
            [Title(Exponential Fog)]
            // ====================================== 指数高度雾参数 ======================================
            [HideInInspector] _ExponentialFogParams0 ("Exponential Fog Param 0 (x:雾层1基准高度; y:雾层1密度; z:雾层1衰减)", Vector) = (0,0.001,0.1,0)
            [HideInInspector] _ExponentialFogParams1 ("Exponential Fog Param 1 (x:雾起始距离; y:起始修正缩放; z:雾结束距离; w:结束修正缩放)", Vector) = (10,0.1,1000,0.001)
            [HideInInspector] _ExponentialFogParams2 ("Exponential Fog Param 2 (xyz:雾颜色; w:最小透射率)", Vector)                = (0.5,0.6,0.7,0.01)
            [HideInInspector] _ExponentialFogParams3 ("Exponential Fog Param 3 (x:雾层2衰减; y:雾层2密度; z:雾层2基准高度)", Vector) = (0.1,0.002,100,0)
        [FoldoutEnd]_FogEnd("_FogEnd", Float) = 0

        [Space(10)][Title(MaterialFlags)]
        [KeysEnum(FLAG_HAIRSHADOW, FLAG_EYELASH, FLAG_HAIRMASK)]
        _ToonFlagsKeywords                          ("ToonFlags", Float)                    = -1
        
        // Other Settings
        [Title(OtherSettings)]
        [Enum(UnityEngine.Rendering.CullMode)] 
        _Cull                                       ("Cull Mode", Float)                    = 2
        [Toggle(_ALPHATEST_ON)]_AlphaClip           ("Alpha Clip", Float)                   = 0
        _Cutoff                                     ("Cutoff", Range(0, 1))                 = 1
        [HideInInspector] _AlphaPremultiply         ("Alpha Premultiply", Float)            = 0
    }
    
    SubShader
    {
        Tags
        {
            "RenderType"="Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue"="Geometry-100"
            "IgnoreProjector" = "True"
            "UniversalMaterialType" = "Character"
        }
        LOD 300

        // GBuffer: write depth and normal
        Pass
        {
            Name "GBufferBase"
            Tags
            {
                "LightMode" = "UniversalGBuffer"
            }

            // -------------------------------------
            // Render State Commands
            ZWrite On
            ZTest LEqual
            Cull [_Cull]

            HLSLPROGRAM
            #pragma target 4.5

            // Deferred Rendering Path does not support the OpenGL-based graphics API:
            // Desktop OpenGL, OpenGL ES 3.0, WebGL 2.0.
            #pragma exclude_renderers gles3 glcore

            // -------------------------------------
            // Shader Stages
            #pragma vertex GBufferPassVertex
            #pragma fragment GBufferPassFragment

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature _ _USE_GPU_ANIMATION _USE_GPU_ANIMATION_STRUCTURED
            #pragma shader_feature_local _ FLAG_HAIRSHADOW FLAG_EYELASH FLAG_HAIRMASK FLAG_FACE
            #pragma shader_feature_local _ALPHATEST_ON

            // -------------------------------------
            // Universal Pipeline keywords
            // #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            //#pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            //#pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION
            // #pragma multi_compile_fragment _ _SHADOWS_SOFT
            // #pragma multi_compile_fragment _ _DBUFFER_MRT1 _DBUFFER_MRT2 _DBUFFER_MRT3
            // #pragma multi_compile_fragment _ _RENDER_PASS_ENABLED
            // #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/RenderingLayers.hlsl"

            // -------------------------------------
            // Unity defined keywords
            // #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            // #pragma multi_compile _ SHADOWS_SHADOWMASK
            // #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            // #pragma multi_compile _ LIGHTMAP_ON
            // #pragma multi_compile _ DYNAMICLIGHTMAP_ON
            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE
            #pragma multi_compile_fragment _ _GBUFFER_NORMALS_OCT

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer
            #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DOTS.hlsl"

            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/UnityGBuffer.hlsl"

            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DeclareDepthTexture.hlsl"

            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Material/PBRToon/PBRToon.hlsl"
            

            #if defined(_USE_GPU_ANIMATION_STRUCTURED)
            StructuredBuffer<float4> _BoneMatrices;
            float   _BoneMatricesCount;
            int4    _BoneIndexOffset;
            int     _BoneMaxCount;
            #endif

            float _GPU_Animation_Tint;
            float _GPU_Animation_Structured_Tint;
            
            #if defined(_USE_GPU_ANIMATION)
            #define MAX_BONE_MATRIX_COUNT 768
            float4 _BoneMatrices[MAX_BONE_MATRIX_COUNT];
            #endif
            
            #if defined(_ALPHATEST_ON)
            float4  _BaseMap_ST;
            float   _Cutoff;

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            #endif

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
            
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Material/UmiToon/EndField/Helper/GPU_Animation_Function.hlsl"

            struct Attributes 
            {
                float4 vertex       :POSITION;
                float3 normal       :NORMAL;
                float4 tangent      :TANGENT;
                float2 uv0          :TEXCOORD0;
                float2 uv1          :TEXCOORD1;
                float2 uv4          :TEXCOORD2;
                float2 uv5          :TEXCOORD3;
                float2 uv6          :TEXCOORD4;
                float2 uv7          :TEXCOORD5;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct Varyings 
            {
                float4 positionHCS      :SV_POSITION;
                float3 positionWS       :TEXCOORD0;
                float3 normalWS         :TEXCOORD1;
                float3 tangentWS        :TEXCOORD2;
                float3 biTangentWS      :TEXCOORD3;
                float2 uv               :TEXCOORD5;
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
            };


            Varyings GBufferPassVertex(Attributes v)
            {
                Varyings o = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                
                ApplyToGpuAnimation(v);
                
                o.positionHCS = TransformObjectToHClip(v.vertex.xyz);
                // o.positionHCS = TransformWorldToHClip(v.vertex.xyz);
                o.positionWS = TransformObjectToWorld(v.vertex.xyz);
                // o.positionWS = v.vertex.xyz;

                o.normalWS = TransformObjectToWorldNormal(v.normal);
                o.tangentWS = TransformObjectToWorldDir(v.tangent.xyz);
                o.biTangentWS = cross(o.normalWS,o.tangentWS) * v.tangent.w * GetOddNegativeScale();

                o.uv = v.uv0;

                return o;
            }

            // We only output normal.
            void GBufferPassFragment(Varyings i
                , out float4 outGBuffer0 : SV_Target0
                #if defined(FLAG_EYELASH)
                , out float4 outGBuffer1 : SV_Target1
                #endif
                , out float4 outGBuffer2 : SV_Target2)
            {
                UNITY_SETUP_INSTANCE_ID(i);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                #if defined(_ALPHATEST_ON)
                float alpha = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv.xy * _BaseMap_ST.xy + _BaseMap_ST.zw).a;
                AlphaDiscard(alpha, _Cutoff);
                #endif

                float3 packedNormalWS = PackNormal(i.normalWS);

                uint toonFlags = 0;
                #if defined(FLAG_HAIRSHADOW)
                {
                    toonFlags |= kToonFlagHairShadow;
                }
                #elif defined(FLAG_EYELASH)
                {
                    toonFlags |= kToonFlagEyelash;
                }
                #elif defined(FLAG_HAIRMASK)
                {
                    toonFlags |= kToonFlagHairMask;
                }
                #elif defined(FLAG_FACE)
                {
                    toonFlags |= kToonFlagFace;
                }
                #endif

                outGBuffer0 = float4(0, 0, 0, EncodeToonFlags(toonFlags));
                outGBuffer2 = float4(packedNormalWS, 0);

                #if defined(FLAG_EYELASH)
                outGBuffer1 = EncodeDepthToRGBA(i.positionHCS.z);
                #endif
            }
            ENDHLSL

        }

        // CharacterForward: shading
        Pass
        {
            Name "CharacterForward"
            Tags
            {
                "LightMode" = "CharacterForward"
            }

            // -------------------------------------
            // Render State Commands
            ZWrite Off
            ZTest Equal
            Cull [_Cull]

            HLSLPROGRAM
            #pragma target 4.5

            // -------------------------------------
            // Shader Stages
            #pragma vertex ForwardToonVert
            #pragma fragment ForwardToonFrag

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature _ _USE_GPU_ANIMATION _USE_GPU_ANIMATION_STRUCTURED
            #pragma shader_feature_local _DECAL_UV
            #pragma shader_feature_local _SPECULAR_LINE
            #pragma shader_feature_local _SHADOW_RAMP
            #pragma shader_feature_local _INDIR_CUBEMAP
            #pragma shader_feature_local _CHARACTER_VFX_SPECIAL
            // We use predepth in gbuffer, no need to do alpha test in CharacterForward
            // #pragma shader_feature_local _ALPHATEST_ON

            // -------------------------------------
            // Universal Pipeline keywords
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            // #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile _ _PEROBJECT_SCREEN_SPACE_SHADOW
            #pragma multi_compile _ _RAYTRACING_SHADOWS
            #pragma multi_compile _ _GPU_LIGHTS_CLUSTER
            #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
            #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            // #pragma multi_compile_fragment _ _DBUFFER_MRT1 _DBUFFER_MRT2 _DBUFFER_MRT3
            #pragma multi_compile_fragment _ _LIGHT_COOKIES
            #pragma multi_compile _ _LIGHT_LAYERS
            #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/RenderingLayers.hlsl"

            // -------------------------------------
            // Unity defined keywords
            // #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            // #pragma multi_compile _ SHADOWS_SHADOWMASK
            // #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            // #pragma multi_compile _ LIGHTMAP_ON
            // #pragma multi_compile _ DYNAMICLIGHTMAP_ON
            // #pragma multi_compile _ USE_LEGACY_LIGHTMAPS
            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE
            #pragma multi_compile_fragment _ _GBUFFER_NORMALS_OCT

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer
            #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DOTS.hlsl"

            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/UnityInstancing.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/UnityGBuffer.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DeclareDepthTexture.hlsl"

            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/GPUCulledLights.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/PreIntegratedFGD.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/PerObjectShadows.hlsl"

            

            #if defined(_USE_GPU_ANIMATION_STRUCTURED)
            StructuredBuffer<float4> _BoneMatrices;
            #endif

            StructuredBuffer<float4> _FragData;
            
            CBUFFER_START(UnityPerMaterial)
            //Debug
            float   _ShowAlbedo;
            
            //GPU Skinning
            #if defined(_USE_GPU_ANIMATION)
                #define MAX_BONE_MATRIX_COUNT 768
                float4 _BoneMatrices[MAX_BONE_MATRIX_COUNT];
            #endif

            float   _GPU_Animation_Tint;
            float   _GPU_Animation_Structured_Tint;
            float   _BoneMatricesCount;
            int4    _BoneIndexOffset;
            int     _BoneMaxCount;

            //Frag Data
            int     _FragDataCount;

            //decode
            float   _SurfaceType;
            float   _BackFaceNormalFlip;
            float   _ShadowColorBrightness;
            float   _ShadowColorSaturation;

            float   _SpecRampIridescentMode;

            float4  _VFXParams0;

            float4  _ExposureParams;
            float4  _CharacterParams0;
            float4  _CharacterParams1;
            float4  _CharacterParams2;
            float4  _CharacterParams3;
            float4  _CharacterParams4;
            float4  _CharacterParams5;
            float4  _CharacterParams6;
            float4  _CharacterParams7;
            float4  _CharacterParams10;
            float4  _EnvironmentGlobalParams0;
            float4  _IVParam0;
            float4  _IVParam1;

            float4  _IVDefaultSHAr;           
            float4  _IVDefaultSHAg;           
            float4  _IVDefaultSHAb;           

            //Frag Params
            float4  _CharacterParams8;
            float4  _CharacterParams9;

            //Base Properties
            float4  _BaseColor;
            float4  _BaseMap_ST;
            float4  _LineMap_ST;
            float   _NormalScale;
            float   _SpecBumpScale;
            float4  _VFXSpecialMainTex_ST;
            float4  _VFXSpecialBlendTex_ST;

            //Raining
            float   _RainFilpObjectDir;
            float4  _cb0_170;
            float  _RainEffectIntensity;
            float  _WetEffectWorldSpaceHeight;
            float  _WetEffectIntensity;
            
            //顶点着色器
            float   _SinEnable;

            // PBR Properties
            float   _Metallic;
            float   _Roughness;
            float   _Occlusion;

            //Anisotropy
            float   _AnisotropyValue;    
            float   _AnisotropyDirX;
            float   _AnisotropyIntensity;
            float   _AnisotropyEdgeFade;
            float   _AnisotropyValue2;   
            float   _AnisotropyRange2;   
            float4  _AnisotropyColor2;

            //Specular Line
            float   _UseLineMap;
            float   _LineAmount;
            float   _LineValue;
            float   _LineRange;
            float   _LineIntensity;
            float   _LineSaturation;

            // Stockings
            float   _AnisoOffset;
            float   _StockingsPow;
            float4  _StockingsColorInside;
            float4  _StockingsColorOutside;

            // Shining Decal
            float   _DecalAnisoUIntensity;
            float   _DecalMaskPower;
            float   _DecalMaskMin;
            float   _DecalMaskMax;

            float   _FwidthRange;

            float   _DecalUVOffset;
            float   _DecalUVScale;

            float   _DecalRoughnessScale;
            float   _DecalMetallicScale;
            
            float   _DecalCenterIntensity;
            float   _DecalRimIntensity;
            
            // Direct Light
            float4  _SelfLight;
            float   _MainLightColorLerp;
            float4  _SelfAddLightColor;
            float   _AddLightColorLerp;
            float   _DirectOcclusion;

            // Shadow
            float4  _DirectionalShadowParams;
            float4  _DirectionalShadowParams2;
            float4  _ShadowColor;
            float   _ShadowOffset;
            float   _ShadowSmoothNdotL;
            float   _ShadowSmoothScene;
            float   _ShadowStrength;

            // Indirect
            float   _EnableLaser;
            float4  _LaserDefaultColor;
            float   _LaserIntensity;
            float   _SimplerSH;
            float4  _SelfEnvColor;
            float   _EnvColorLerp;
            float   _IndirDiffUpDirSH;
            float   _IndirDiffIntensity;
            float   _IndirSpecCubeWeight;
            float   _IndirSpecIntensity;

            // Emission
            float   _EmissionIntensity;
            float4  _EmissionCol;

            //VFX Special
            float   _VFXMainUVSet;
            
            float   _VFXFresnelBias;
            float   _VFXFresnelAffectOpacity;
            float   _VFXFresnelPower;
            float   _VFXFresnelFlip;
            
            float   _UseVFXMainTexAsAlpha;
            float   _SpecialDissolveScheduleOffset;
            float   _VFXColorIntensity;
            float   _VFXColorAlpha;
            
            float4  _VFXSpecialParam;
            float4  _VFXBlendTint;          
            float4  _VFXFresnelColor;
            float4  _VFXColor;

            // RimLight
            float4  _RimLightPosWS;
            float4  _DirectRimFrontCol;
            float4  _DirectRimBackCol;
            float   _DirectRimWidth;
            float   _PunctualRimWidth;

            //FOG
            float4  _FogColor_Height;        
            float4  _FogColor_Directional;   
            float4  _FogColor_Transition;    
            
            float4  _FogColor_MixE;          
            float4  _FogColor_MixF;          
            
            float   _FogBaseDensity;              
            float   _FogMieK;                     
            float   _FogHeightBias;               
            float   _FogDistanceOffset;           
            float   _FogHeightRange;              
            float4  _cb0_136;                     
            
            //extra fog
            float4  _CharacterParams11;

            float4  _AtmosphereFogParams0; 
            float4  _AtmosphereFogParams1; 
            float4  _AtmosphereFogParams2; 
            float4  _AtmosphereFogParams3; 
            float4  _AtmosphereFogParams4; 
            float4  _AtmosphereFogParams5; 

            float4  _VolumetricFogParams0; 
            float4  _VolumetricFogParams1; 
            float4  _VolumetricFogParams2; 
            float4  _VolumetricFogParams3; 
            float4  _VolumetricFogParams4; 
       
            float4  _ExponentialFogParams0;
            float4  _ExponentialFogParams1;
            float4  _ExponentialFogParams2;
            float4  _ExponentialFogParams3;

            // Alpha Test
            float   _Cutoff;
            float   _AlphaPremultiply;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            TEXTURE2D(_PBRMask);
            SAMPLER(sampler_PBRMask);

            TEXTURE2D(_LineMap);
            SAMPLER(sampler_LineMap);

            TEXTURE2D(_NormalMap);
            SAMPLER(sampler_NormalMap);

            TEXTURE2D(_SplitNormalMap);
            SAMPLER(sampler_SplitNormalMap);

            // TEXTURE2D(_DecalMap);
            // SAMPLER(sampler_DecalMap);

            TEXTURE2D(_EmissionMap);
            SAMPLER(sampler_EmissionMap);

            TEXTURE2D(_CharacterRainEffectTex);
            SAMPLER(sampler_CharacterRainEffectTex);
            TEXTURE2D(_CharacterRainStreakTex);
            SAMPLER(sampler_CharacterRainStreakTex);

            TEXTURE2D(_VFXSpecialMainTex);
            SAMPLER(sampler_VFXSpecialMainTex);
            
            TEXTURE2D(_VFXSpecialBlendTex);
            SAMPLER(sampler_VFXSpecialBlendTex);

            TEXTURE2D(_LaserMap);
            SAMPLER(sampler_LaserMap);

            TEXTURE3D(_T3);
            TEXTURE3D(_T4);
            TEXTURE3D(_T5);
            // SAMPLER(sampler_T3);

            TEXTURE2D(_ShadowRampTex);
            SAMPLER(sampler_ShadowRampTex);

            TEXTURE2D(_SpecRampMap);
            SAMPLER(sampler_SpecRampMap);

            TEXTURECUBE(_IndirSpecCubemap);

            TEXTURE3D(_IntegratedLightScattering);

            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Material/UmiToon/EndField/EndFieldToonFunc/EndFieldToonHairFunc.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Material/UmiToon/EndField/Helper/GPU_Animation_Function.hlsl"

            Varyings ForwardToonVert(Attributes v)
            {
                Varyings o;
                ZERO_INITIALIZE(Varyings, o);
                
                UNITY_SETUP_INSTANCE_ID(v); 
                UNITY_TRANSFER_INSTANCE_ID(v,o); 
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                ApplyToGpuAnimation(v);
                
                real3 posWS = TransformObjectToWorld(v.vertex.xyz);
                real3 posVS = TransformWorldToView(posWS);
                real4x4 matrix_offset = {
                    1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f
                };
                float4x4 UNITY_MATRIX_P_offset = mul(UNITY_MATRIX_V, UNITY_MATRIX_P);
                float4x4 UNITY_MATRIX_VP_offset = mul(UNITY_MATRIX_P, UNITY_MATRIX_V);
                // o.positionHCS = mul(UNITY_MATRIX_VP_offset, float4(posWS, 1.0));
                // o.positionHCS = mul(GetViewToHClipMatrix(), float4(posVS, 1.0));
                // o.positionHCS = TransformWViewToHClip(posVS);
                // o.positionHCS = TransformWorldToHClip(posWS);

                o.positionOS = v.vertex.xyz;
                o.positionWS = TransformObjectToWorld(v.vertex.xyz);
                o.positionHCS = TransformObjectToHClip(v.vertex.xyz);
                
                o.normalOS = v.normal;
                o.normalWS.xyz = TransformObjectToWorldNormal(v.normal, true);
                o.tangentWS.xyz = TransformObjectToWorldDir(v.tangent.xyz, true);
                o.tangentWS.w = v.tangent.w;
                // o.positionHCS = TransformWorldToHClip(v.vertex.xyz);
                // o.positionWS = v.vertex.xyz;
                // o.normalWS.xyz = normalize(v.normal);
                // o.tangentWS.xyz = normalize(v.tangent.xyz);

                o.bitangentWS.xyz = cross(o.normalWS.xyz, o.tangentWS.xyz) * v.tangent.w * GetOddNegativeScale();
                
                o.viewDirWS.xyz = /*lerp(*/GetWorldSpaceNormalizeViewDir(o.positionWS)/*, normalize(UNITY_MATRIX_V[2].xyz), _cb0_66.w)*/;

                
                o.color = v.color;
                o.uv.xy = TRANSFORM_TEX(v.uv0.xy, _BaseMap);
                o.uv.zw = v.uv1.xy;

                return o;
            }


            float4 ForwardToonFrag(Varyings i, uint facing : VFACE) : SV_Target0
            {
                UNITY_SETUP_INSTANCE_ID(i);
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);

                EndFieldSurface surface;
                ZERO_INITIALIZE(EndFieldSurface, surface);
                IntializeEndFieldSurface(i, facing, surface);
                
                //debug
                if (_ShowAlbedo) return float4(surface.basecolor, 1);
                
                uint meshRenderingLayers = GetMeshRenderingLayer();

                Lighting directLighting;
                Lighting indirectLighting;
                ZERO_INITIALIZE(Lighting, directLighting);
                ZERO_INITIALIZE(Lighting, indirectLighting);

                Light mainLight = GetMainLight();
                EndFieldLightData lightData;
                ZERO_INITIALIZE(EndFieldLightData, lightData);
                InitializeEndFieldLightData(mainLight, lightData);

                //region 环境diffuse SH Color
                EndFieldEnvData envData;
                ZERO_INITIALIZE(EndFieldEnvData, envData);
                ApplyEnvFeature(surface, envData);
                
                EndFieldShadowData shadowData;
                ZERO_INITIALIZE(EndFieldShadowData, shadowData);
                IntializeEndFieldShadow(surface, shadowData);

                EndFieldVecData vecData;
                ZERO_INITIALIZE(EndFieldVecData, vecData);
                InitializeEndFieldVecData(i, lightData, shadowData, surface, vecData);

                CalculateSpecularBitangent(vecData, surface);
                
                float NdotUp_clamp = saturate(_CharacterParams7.x + dot(surface.normalWS, _CharacterParams6.xyz)) 
                                    * _CharacterParams7.y + _CharacterParams7.z;     //计算顶光吗
                //region 湿身feature
                ApplyRainFeature(surface);

                EndFieldDotData dotData;
                ZERO_INITIALIZE(EndFieldDotData, dotData);
                InitializeEndFieldDotData(vecData, surface, dotData);
                
                //region diffuse calculate
                EndFieldBSDF bsdf;
                ZERO_INITIALIZE(EndFieldBSDF, bsdf);
                SurfaceConvertToBSDF(vecData, dotData, surface, bsdf);

                float MaxRampSubMinRamp = Max3(bsdf.ramp_NdotL.x, bsdf.ramp_NdotL.y, bsdf.ramp_NdotL.z)
                                        - Min3(bsdf.ramp_NdotL.x, bsdf.ramp_NdotL.y, bsdf.ramp_NdotL.z);
                
                //综合阴影项
                float occlusionShadow = shadowData.char * surface.occlusion;  //只是阴影和环境光遮蔽
                float ramp2ShadowAtten = bsdf.ramp_NdotV * occlusionShadow;   //补充NdotV的菲涅尔细节
                float shadowMixFactor = saturate(ramp2ShadowAtten + bsdf.ramp_NdotL.a);   //补充NdotL的光照渐变细节

                // 混合基础 Diffuse (Shadow part)
                float3 diffuseShadowColor = lerp(bsdf.diffuse_shadow2, bsdf.diffuse_shadow * _CharacterParams0.z, shadowMixFactor);       //r20.xyz

                // 计算最暗阴影因子
                float ramp1minShadowFactor = Min3(shadowData.char, surface.occlusion, bsdf.ramp_NdotL.a);
                // 混合最终 Diffuse Base
                float3 diffuseBase = lerp(diffuseShadowColor, bsdf.diffuse, ramp1minShadowFactor);
                // 应用 Ramp 染色, 将染色ramp添加到明暗交界线部分
                float3 ramp_diffuseColor = lerp(1, bsdf.ramp_NdotL.xyz, MaxRampSubMinRamp) * diffuseBase;
                // 将ramp后过重的颜色明暗系数清除，只保留颜色变化
                float rampGray = max(0.001, Luminance(ramp_diffuseColor));
                float3 avgvalue_ramp_diffuseColor = ramp_diffuseColor * clamp(Luminance(diffuseBase) * rcp(rampGray), 0, 1.5);
                
                // 7. 间接光与高光计算 (Indirect & Specular)
                // 限制 Ramp Scale
                float3 scaleClamped;
                scaleClamped.x = clamp(envData.envIntensity,  0,    1.5);
                scaleClamped.y = clamp(envData.envIntensity,  1.25, 1.75);
                scaleClamped.z = clamp(envData.envIntensity,  0.5,  1.5);

                // 间接光基础
                float indSpecMask = ramp1minShadowFactor * min(1, _CharacterParams1.y);
                float3 indSpecBase = NdotUp_clamp * lerp(envData.envColor_fixed, 1, indSpecMask);
                
                // 混合直射光颜色，降低暗面区域饱和度
                float3 saturatedDirectionalLightColor = lerp(Luminance(lightData.color), lightData.color, ramp1minShadowFactor);

                float3 indSpecStep1 = scaleClamped.x * indSpecBase * lerp(1.0f, lightData.standardColor, lightData.useStandardColor);
                indSpecStep1 = indSpecStep1 + saturatedDirectionalLightColor;

                // 混合最终间接高光
                float scaleSmooth = min(envData.envIntensity * 0.35f + 0.65f, 1.5f);
                float finalSpecScale = lerp(scaleSmooth, scaleClamped.y, _CharacterParams1.x);  //切换另一种钳制曲线

                float3 indSpecStep2 = indSpecBase * finalSpecScale * _CharacterParams0.w;
                float3 combineLightColor = lerp(indSpecStep2, indSpecStep1 * _CharacterParams0.y, shadowData.scene);

                // 8. 最终 Diffuse 输出 (Final Diffuse)
                // 增加饱和度
                float3 diffuseSat = lerp(Luminance(bsdf.diffuse), bsdf.diffuse, 1.2f);

                // 混合最终 Diffuse
                float3 diffuseFinalStep1 = lerp(bsdf.diffuse_shadow * _CharacterParams0.z, diffuseSat, ramp2ShadowAtten);
                float3 sharpenDiffuse = lerp(diffuseFinalStep1, avgvalue_ramp_diffuseColor, shadowData.scene);  //提高对比度，减少暗面，增加投影与非投影区域的对比
                
                float3 diffuseLighting = combineLightColor * sharpenDiffuse;

                // region anisotropy specular calculate
                // 阴影处spec的强度权重调整
                float shadowLerp = lerp(ramp2ShadowAtten, ramp1minShadowFactor, shadowData.scene);
                float specShadowWeight = lerp(_CharacterParams0.z, 1, shadowLerp) * (shadowLerp * 0.5 + 0.5);

                //==================== 第一层高光 =======================
                float3 specColor_small = Hair_D_EndField(vecData, surface);
                float specMask_small = Max3(specColor_small.x, specColor_small.y, specColor_small.z);

                //================  第二层高光  ========================
                float3 specColor_large = Hair_F_EndField(vecData, surface);

                //region line map
                float lineAmount = ceil(max(frac(surface.baseUV.x * _LineAmount) - 0.5f, 0.0f));

                float lineValue = _LineValue * 2.0f - 1.0f;
                float3 anisoTangent_final = ShiftTangent(surface.specBitangent, surface.normalWS_smooth, lineValue);
                float anisoSpec_final = AnisotropicSpecular(anisoTangent_final.xyz, vecData.halfDirWS);

                float lineTemp = lerp(lineAmount, 1.0f - surface.hairLine, _UseLineMap);
                float lineIntensity = lerp(1.0f - _LineIntensity, 1.0f, lineTemp);
                float lineIntensity_mask = lerp(lineIntensity, 1.0f, specMask_small);
                float power_anisoSpec_final = trunc(max(1.0f  - _LineRange, 0.0f) * 200.0f);
                float _2018 = mad(surface.specularLevel, (1.0f - lineIntensity_mask) * pow(anisoSpec_final, power_anisoSpec_final), 1.0f);
                float lineSaturation = lerp(_LineSaturation, 1.0f, _2018);

                //combine Color
                float3 someDiffuseColor = diffuseLighting * _2018;

                // 最终 Alpha
                float alpha = lerp(1, surface.alpha, _AlphaPremultiply);

                float3 sharpenDiffuse_final = lerp(Luminance(someDiffuseColor), someDiffuseColor, lineSaturation);

                float3 spec_large = (1.0f - specMask_small) * specColor_large * surface.rainMask;
                float3 spec_small = bsdf.specularLevel * specColor_small * _AnisotropyIntensity * 5.0f * surface.rainMask;

                float3 directSpecular = combineLightColor * specShadowWeight * (spec_small.xyz + spec_large.xyz);
                float3 lightingColor = sharpenDiffuse_final.xyz * alpha + directSpecular;

                float gray_lightingColor = Luminance(lightingColor);
                float gray_lightingColor_clamp = clamp(gray_lightingColor - 0.5, 0, 0.5);
                gray_lightingColor_clamp = gray_lightingColor_clamp * gray_lightingColor_clamp + 1;

                float3 mainLightingColor = lerp(gray_lightingColor, lightingColor, gray_lightingColor_clamp);
                
            //region distance rim light calculate
                float3 rimColor = ApplyRimFeature(diffuseSat, vecData, shadowData, surface);
            //region env rim light
                float3 SHRimColor = ApplySHRimFeature(diffuseSat, saturatedDirectionalLightColor, vecData, shadowData, dotData,
                    envData, surface, bsdf);

        
            //region combine lighting
            //==================== combine lighting ====================
                float3 finalColor = mainLightingColor;
                // finalColor += indirectSpecular * envData.envColor_fixed;
                finalColor += SHRimColor;
                // finalColor += effectColor * alpha;
                finalColor += rimColor;

            // return float4(finalColor, 1);

                //region additonal light part
                float3 color_r13 = finalColor;
                //多光源部分
                // uint lightCount = 0;
                // while (true)
                // {
                //     if (7 < lightCount)
                //         break;
                //     finalShadowAtten_inv = x0[lightCount + 0].x;
                //     r3_w = (uint)lightCount << 5;
                //     float3 color_r14 = color_r13;

                //     // r4_w = finalShadowAtten_inv;
                //     // while (true)
                //     // {
                //     //     if (r4_w == 0)
                //     //         break;
                //     //     
                //     // }

                //     color_r13 = color_r14;
                //     lightCount = lightCount + 1;
                // }
                    
                float3 resultColor = color_r13 / _ExposureParams.xxx;
                float4 outputColor;
                outputColor.a = _SurfaceType == 1.0f ? surface.alpha : 1.0f;
                //region fog
                if (_CharacterParams11.w < 0.5) 
                {
                    // =========================================================
                    // Part A: 几何与大气散射准备 (Atmosphere Pre-calculation)
                    // =========================================================
                    // --- 计算视线向量 (处理正交/透视)
                    float3 viewVec = GetWorldSpaceNormalizeViewDir(i.positionWS.xyz);
                    float distSq = dot(viewVec, viewVec);
                    // 防止距离为0导致的NaN
                    float invDist = rsqrt(max(distSq, 1e-8)); 
                    float dist = distSq * invDist;
                    float3 viewDir = viewVec * invDist; // Normalized
                    // 归一化视线方向 (指向像素/远离相机)
                    float3 viewDirFromCam = -viewDir;

                    // -----------------------------------------------------------
                    // 2. 大气散射参数计算 (Atmospheric Scattering)
                    // -----------------------------------------------------------
                    // 算法参考: 基于高度的 Rayleigh/Mie 散射积分

                    // _AtmosphereFogParams5.xyz 是主光源(太阳)方向
                    float cosTheta = dot(viewDirFromCam, _AtmosphereFogParams5.xyz);

                    // 计算相对于海平面的高度 (单位转换: mm -> m 或类似缩放)
                    // _AtmosphereFogParams2.w 是 Base Height
                    float relHeight = i.positionWS.y * 0.001f - _AtmosphereFogParams2.w;

                    // 计算光学深度 (Optical Depth) 的积分近似
                    // _AtmosphereFogParams4.w = Max Height
                    // _AtmosphereFogParams0.w = H (Scale Height, 标高)
                    float heightDiffNorm = max((_AtmosphereFogParams4.w - _AtmosphereFogParams2.w - relHeight) 
                                            / _AtmosphereFogParams0.w, 0.01f);

                    // 基于 Beer-Lambert 定律的透射率计算
                    // 这里的数学是求解 exp(-h/H) 沿着视线的积分
                    float extinctionCoeff = exp2(heightDiffNorm * -K_LOG2E); // exp(-x)

                    // 这里涉及视线方向的投影计算，用于确定积分的路径长度比例
                    // sqrt(somePosDir2) 是距离，_2968/_2970/_2972 是方向向量分量
                    // _AtmosphereFogParams5.w 和 3.w 参与计算几何路径长度
                    float pathLengthScale = max(sqrt(distSq) * _AtmosphereFogParams5.w 
                                            - _AtmosphereFogParams3.w, 0.0f);

                    // 计算积分上限的指数项
                    // 1.442695 = log2(e)，将 exp 转换为 exp2
                    float expTop = exp2(heightDiffNorm * (-1.44269502f));

                    // 计算积分下限的指数项 (当前位置)
                    float expBottom = exp2((-relHeight / _AtmosphereFogParams0.w) * 1.44269502f);

                    // 代表了路径上的累积密度 (Optical Depth)
                    // 解析积分结果
                    // 公式对应: (exp(-Height_Top) - exp(-Height_Bottom)) / Slope * Length
                    // 这里的 (-get_a) 是路径长度因子
                    float opticalDepth = (-pathLengthScale) * ((1.0f - expTop) / heightDiffNorm) * expBottom;

                    // 计算三通道的大气透射率 (Transmittance)
                    // AtmosphereFogParams1/3 是散射系数 (Scattering Coefficients)
                    // scatteringCoeff是预计算好的散射系数 (Rayleigh + Mie + Absorption)
                    float3 atmosphereTransmittance;
                    float3 scatteringCoeff = _AtmosphereFogParams1.xyz + _AtmosphereFogParams3.xyz + _AtmosphereFogParams2.xyz; // 组合各层系数
                    atmosphereTransmittance.x = exp2(scatteringCoeff.x * opticalDepth * K_LOG2E);
                    atmosphereTransmittance.y = exp2(scatteringCoeff.y * opticalDepth * K_LOG2E);
                    atmosphereTransmittance.z = exp2(scatteringCoeff.z * opticalDepth * K_LOG2E);

                    // -----------------------------------------------------------
                    // 3. 米氏散射相位函数 (Mie Phase Function)
                    // -----------------------------------------------------------
                    // 模拟看向太阳时的光晕
                    // 近似于 Henyey-Greenstein Phase Function
                    float miePhase = mad(cosTheta, cosTheta, 1.0f) * 0.0596831f; // 常数可能与 1/4pi 有关
                    // 计算相位函数的各项异性部分 (g值)
                    // CB0[132].w 是 Mie Anisotropy (g)
                    float g = _AtmosphereFogParams1.w;
                    float mieDenominator = 1.0f + g * g - 2.0f * g * cosTheta;
                    float mieScattering = (1.0f - g * g) / (pow(mieDenominator, 1.5) * 4.0f * K_PI); // 标准 HG 公式

                    // 公式大致为: SkyColor = (Rayleigh * PhaseR + Mie * PhaseM) * SunIntensity
                    // Text 1: mad(MieCoeff, Phase, RayleighCoeff * Phase2)
                    float3 factor = _AtmosphereFogParams3.xyz * miePhase + mieScattering * _AtmosphereFogParams1.xyz;

                    // 大气散射产生的 In-Scattering 颜色 (get_temp_x/y/z)
                    // 基于透射率和相位函数计算出的天空背景色
                    // 应用光源颜色/强度 (_AtmosphereFogParams0)    可能是太阳光颜色强度
                    // _AtmosphereFogParams4 可能是环境光项
                    float3 atmosphereInScatter = _AtmosphereFogParams0.xyz * factor 
                                + (_AtmosphereFogParams1.xyz + _AtmosphereFogParams3.xyz) * _AtmosphereFogParams4.xyz;

                    // 除以散射总系数 (_3039/_3040...) 归一化，并 Clamp 防止溢出
                    // 这步对应物理公式中的: Integral(L_sun * rho * exp(-T)) -> Result / Extinction
                    atmosphereInScatter /= scatteringCoeff;
                    atmosphereInScatter = clamp(atmosphereInScatter, 0.0f, 255.0f); // HDR 范围限制
                                
                    // 步骤 A: 混合大气散射背景 (Physically Based Sky)
                    // 公式: Result = AtmosphereInScatter + SceneColor * AtmosphereTransmittance
                    float3 colorWithAtmosphere;
                    colorWithAtmosphere.x = lerp(atmosphereInScatter.x, resultColor.x, atmosphereTransmittance.x);
                    colorWithAtmosphere.y = lerp(atmosphereInScatter.y, resultColor.y, atmosphereTransmittance.y);
                    colorWithAtmosphere.z = lerp(atmosphereInScatter.z, resultColor.z, atmosphereTransmittance.z);

                    float3 finalFogColor = float3(0,0,0);
                    float finalFogOpacity = 0.0f; // 1.0 = 全雾, 0.0 = 无雾

                    if (_VolumetricFogParams0.z > 0.0f)
                    {
                        // === 体积雾路径 (Volumetric Fog Path) ===
                        // 算法参考: Assassin's Creed 4 (Siggraph 2014) - 3D Texture Volumetric Fog

                        // A. 计算 Z Slice (对数空间)
                        // inp.position.w 是线性深度 (Linear Depth)
                        float linearDepth = 1 / i.positionHCS.w;
                        float zLog = log(linearDepth * _VolumetricFogParams1.x + _VolumetricFogParams1.y);
                        float zSlice = zLog * _VolumetricFogParams1.z;
                        // 归一化 Z (0-1)
                        float normalizedZ = zSlice / _VolumetricFogParams0.z;

                        // B. 计算抖动 (Jitter) 以消除切片采样的带状伪影
                        // 使用 FrameCount 和屏幕坐标进行空间哈希
                        int frameCount = (int)_Time.y * 60;
                        uint seedY = (uint(surface.screenPos.y) * 1664525u) + 1013904223u;
                        uint seedCnt = ((asuint(frameCount) & 7u) * 1664525u) + 1013904223u;
                        uint seedMix = (seedY * seedCnt) + ((uint(surface.screenPos.x) * 1664525u) + 1013904223u);

                        // 混合哈希并生成 0~1 的随机浮点数
                        uint hash1 = (seedCnt * seedMix) + seedY;
                        uint hash2 = (seedMix * hash1) + seedCnt;
                        uint hash3 = (hash1 * hash2) + seedMix;

                        // [Text 1: _3217 >> 16u] 将高位取出来作为随机数
                        // 3.0518...e-05f 是 1/32768，归一化到 0~1 (或者是 -1~1)
                        // _IntegratedLightScattering 采样坐标计算中用到了这个抖动
                        // CB0[145].w (_VolumetricFogParams4.w) 控制抖动强度
                        float jitterX = float(hash3 >> 16u) * 3.05180437e-05f; 
                        float jitterY = float(((hash2 * hash3) + hash1) >> 16u) * 3.05180437e-05f;

                        float3 uvw;
                        // CB0[143] (_VolumetricFogParams2) 包含纹理尺寸/缩放因子
                        // mad 操作应用了抖动偏移
                        uvw.x = (surface.screenPos.x + jitterX * _VolumetricFogParams4.w) * _VolumetricFogParams2.x;
                        uvw.y = (surface.screenPos.y + jitterY * _VolumetricFogParams4.w) * _VolumetricFogParams2.y;
                        uvw.z = normalizedZ; // 通常 Z 轴不需要 XY 平面那样的抖动，或者已经在 Slice 计算中隐含


                        // -------------------------------------------------------------------------
                        // 2. 计算积分射线的几何参数 (Geometry Setup)
                        // -------------------------------------------------------------------------
                        // 获取视线向量和相机参数
                        float3 camForward = float3(_ViewMatrix[0].z, _ViewMatrix[1].z, _ViewMatrix[2].z); // 假设提取前向

                        // 计算视线与相机Z轴的夹角余弦 cos(theta)
                        float cosTheta = dot(-viewDir, -camForward);

                        // 计算射线长度 (Ray Length)
                        // 射线长度 = 体积雾截断深度 / cos(theta)
                        // 这确立了积分的终点：视线穿出体积雾包围盒的那一点
                        float volMaxZ = _VolumetricFogParams0.w;
                        float rayLength = (cosTheta > 1e-8f) ? (volMaxZ * rcp(cosTheta)) : volMaxZ;

                        // 计算射线终点的高度 (End Height)
                        // EndHeight = CamHeight + RayLength * ViewDir.y
                        float endHeight = GetCurrentViewPosition().y + rayLength * viewDir.y;

                        // -------------------------------------------------------------------------
                        // 3. 在射线段内积分解析高度雾 (Analytical Fog Integration)
                        // -------------------------------------------------------------------------
                        
                        // A. 第一层雾 (Layer 1)
                        // 参数: 射线Y分量 (EndHeight - StartHeight), CamHeight, Base, Falloff, Density
                        float vectorY = endHeight - GetCurrentViewPosition().y; // 其实就是 rayLength * viewDir.y

                        float fogInt0 = ComputeHeightFogIntegral(vectorY, GetCurrentViewPosition().y, 
                                                                _ExponentialFogParams0.x, 
                                                                _ExponentialFogParams0.z, 
                                                                _ExponentialFogParams0.y);
                                                            
                        float fogInt1 = ComputeHeightFogIntegral(vectorY, GetCurrentViewPosition().y, 
                                                                _ExponentialFogParams3.z, 
                                                                _ExponentialFogParams3.x, 
                                                                _ExponentialFogParams3.y);

                        // --- 组合两层雾 ---
                        // 注意：这里积分的是 [Camera, VolumetricEnd] 这一段的解析雾
                        float totalDensity = _ExponentialFogParams3.y > 0.0f ? (fogInt0 + fogInt1) : fogInt0;

                        float opticalDepth = totalDensity * rayLength;// 假设函数内已经处理了长度

                        float anaTransmittance = exp2(-opticalDepth * 1.442695f);
                        anaTransmittance = min(anaTransmittance, 1.0f);
                        anaTransmittance = max(anaTransmittance, _ExponentialFogParams2.w);

                        // D. 距离修正 (Fog Start / End Mask)
                        // 应用于解析雾的范围修正
                        float startMask = saturate((_ExponentialFogParams1.x - dist) * _ExponentialFogParams1.y);
                        float endMask = saturate((dist - _ExponentialFogParams1.z) * _ExponentialFogParams1.w);

                        float finalAnaTrans = min(anaTransmittance + startMask + endMask, 1.0f);
                        float anaOpacity = 1.0f - finalAnaTrans;

                        float3 anaColor = anaOpacity * _ExponentialFogParams2.xyz;

                        // -------------------------------------------------------------------------
                        // E. 采样与混合 (Sampling & Composition)
                        // -------------------------------------------------------------------------
                        float4 volFogSample = _IntegratedLightScattering.SampleLevel(sampler_LinearClamp, uvw, 0.0f);

                        // D. 距离遮罩 (Distance Mask)
                        // 限制体积雾的最大距离，超出部分平滑淡出, 是处理天空盒距离
                        // _VolumetricFogParams3.z = Max Distance
                        float distMask = saturate((linearDepth - _VolumetricFogParams3.z) * 1000000.0f);

                        // --- E. 混合参数准备 ---
                        float4 effectiveVol = lerp(volFogSample, float4(0,0,0,1), distMask);

                        // 输出变量赋值
                        finalFogColor = anaColor * effectiveVol.w + effectiveVol.xyz;
                        finalFogOpacity = finalAnaTrans * effectiveVol.w;
                    }
                    else
                    {
                        float3 toPixelVec = i.positionWS.xyz - GetCurrentViewPosition();
                        float distSqr = dot(toPixelVec, toPixelVec);
                        float invDist = rsqrt(max(distSqr, 0.000000001f));

                        float dist = distSqr * invDist;
                        float3 viewDir = toPixelVec * invDist;

                        // --- 雾层 1 计算 (ExponentialFogParams0) ---
                        // _ExponentialFogParams0: x=BaseHeight, y=Density, z=Falloff
                        float fogInt0 = ComputeHeightFogIntegral(toPixelVec.y, GetCurrentViewPosition().y, 
                                                                _ExponentialFogParams0.x, 
                                                                _ExponentialFogParams0.z, 
                                                                _ExponentialFogParams0.y);

                        // --- 雾层 2 计算 (ExponentialFogParams3) ---
                        // _ExponentialFogParams3: y=Density, x=Falloff, z=BaseHeight
                        float fogInt1 = ComputeHeightFogIntegral(toPixelVec.y, GetCurrentViewPosition().y, 
                                                                _ExponentialFogParams3.z, 
                                                                _ExponentialFogParams3.x, 
                                                                _ExponentialFogParams3.y);

                        // --- 组合两层雾 ---
                        float totalDensity = _ExponentialFogParams3.y > 0.0f ? (fogInt0 + fogInt1) : fogInt0;

                        // --- 计算基础透射率 (Base Transmittance) ---
                        float transmittance = exp2(-totalDensity * 1.442695f);
                        transmittance = min(transmittance, 1.0f);
                        // 限制最大 Opacity (通过最小 Transmittance)
                        transmittance = max(transmittance, _ExponentialFogParams2.w);

                        // 5. 应用距离修正 (Distance Masking / Fog Start)
                        // 看起来 _ExponentialFogParams1 用于定义"无雾区"或"线性衰减区"

                        // 修正项 A: (FogStart - dist) * Scale
                        // 如果 dist 小于 FogStart，startMask > 0，增加透射率（减少雾）
                        float startOffset = _ExponentialFogParams1.x - dist;
                        float startMask = saturate(startOffset * _ExponentialFogParams1.y);

                        // 修正项 B: (dist - ParamZ) * ParamW
                        // 可能是用于天空盒距离的修正，或者另一个距离剔除
                        float endOffset = dist - _ExponentialFogParams1.z;
                        float endMask = saturate(endOffset * _ExponentialFogParams1.w);

                        // 组合修正项
                        float finalTransmittance = min(endMask + startMask + transmittance, 1.0f);

                        // 6. 计算最终 Opacity 和颜色
                        float opacity = 1.0f - finalTransmittance;

                        finalFogColor = opacity * _ExponentialFogParams2.xyz;
                        finalFogOpacity = finalTransmittance;
                    }

                    // -----------------------------------------------------------
                    // 5. 最终合成 (Final Composition)
                    // -----------------------------------------------------------
                    outputColor.xyz = colorWithAtmosphere * finalFogOpacity + finalFogColor;
                } 
                else 
                {
                    outputColor.xyz = resultColor;    //最终颜色输出
                }

                return outputColor;
            }
            ENDHLSL

        }
        
        // Outline
        UsePass "DanbaidongRP/EndFieldToon/Helpers/Outline/ForwardOutline"

        // ShadowCaster: Same as Lit.shader
        Pass
        {
            Name "ShadowCaster"
            Tags
            {
                "LightMode" = "ShadowCaster"
            }

            // -------------------------------------
            // Render State Commands
            ZWrite On
            ZTest LEqual
            ColorMask 0
            Cull[_Cull]

            HLSLPROGRAM
            #pragma target 2.0

            // -------------------------------------
            // Shader Stages
            #pragma vertex ShadowPassVertex
            #pragma fragment ShadowPassFragment

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local _ALPHATEST_ON
            // #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DOTS.hlsl"

            // -------------------------------------
            // Universal Pipeline keywords

            // -------------------------------------
            // Unity defined keywords
            #pragma multi_compile _ LOD_FADE_CROSSFADE

            // This is used during shadow map generation to differentiate between directional and punctual light shadows, as they use different formulas to apply Normal Bias
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/ShadowCasterPass.hlsl"
            ENDHLSL
        }

        // DepthOnly
        Pass
        {
            Name "DepthOnly"
            Tags
            {
                "LightMode" = "DepthOnly"
            }

            // -------------------------------------
            // Render State Commands
            ZWrite On
            ColorMask R
            Cull[_Cull]

            HLSLPROGRAM
            #pragma target 2.0

            // -------------------------------------
            // Shader Stages
            #pragma vertex DepthOnlyVertex
            #pragma fragment DepthOnlyFragment

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local _ALPHATEST_ON
            // #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A

            // -------------------------------------
            // Unity defined keywords
            #pragma multi_compile_fragment _ LOD_FADE_CROSSFADE

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing
            #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DOTS.hlsl"

            // -------------------------------------
            // Includes
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/DepthOnlyPass.hlsl"
            ENDHLSL
        }

    }

    SubShader
    {
        Tags{ "RayTracingRenderPipeline" = "DanbaidongRP" }
        Pass
        {
            Name "IndirectDXR"
            Tags{ "LightMode" = "IndirectDXR" }

            HLSLPROGRAM

            // -------------------------------------
            // Shader Stages
            #pragma only_renderers d3d11 xboxseries ps5
            #pragma raytracing surface_shader

      
            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local _NORMALMAP
            #pragma shader_feature_local _PARALLAXMAP
            #pragma shader_feature_local _RECEIVE_SHADOWS_OFF
            #pragma shader_feature_local _ _DETAIL_MULX2 _DETAIL_SCALED
            #pragma shader_feature_local_fragment _SURFACE_TYPE_TRANSPARENT
            #pragma shader_feature_local _ALPHATEST_ON
            #pragma shader_feature_local_fragment _ _ALPHAPREMULTIPLY_ON _ALPHAMODULATE_ON
            #pragma shader_feature_local_fragment _EMISSION
            #pragma shader_feature_local_fragment _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
            #pragma shader_feature_local_fragment _OCCLUSIONMAP
            #pragma shader_feature_local_fragment _SPECULARHIGHLIGHTS_OFF
            #pragma shader_feature_local_fragment _ENVIRONMENTREFLECTIONS_OFF
            #pragma shader_feature_local_fragment _SPECULAR_SETUP

            // -------------------------------------
            // Universal Pipeline keywords
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            // #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            // #pragma multi_compile _ EVALUATE_SH_MIXED EVALUATE_SH_VERTEX
            // #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION
            // #pragma multi_compile_fragment _ _SHADOWS_SOFT _SHADOWS_SOFT_LOW _SHADOWS_SOFT_MEDIUM _SHADOWS_SOFT_HIGH
            // #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
            // #pragma multi_compile_fragment _ _DBUFFER_MRT1 _DBUFFER_MRT2 _DBUFFER_MRT3
            // #pragma multi_compile_fragment _ _LIGHT_COOKIES
            // #pragma multi_compile _ _LIGHT_LAYERS
            // #pragma multi_compile _ _FORWARD_PLUS
            // #include_with_pragmas "Packages/com.unity.render-pipelines.core/ShaderLibrary/FoveatedRenderingKeywords.hlsl"
            // #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/RenderingLayers.hlsl"


            // -------------------------------------
            // Unity defined keywords
            // #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            // #pragma multi_compile _ SHADOWS_SHADOWMASK
            // #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            // #pragma multi_compile _ LIGHTMAP_ON
            // #pragma multi_compile _ DYNAMICLIGHTMAP_ON
            // #pragma multi_compile _ USE_LEGACY_LIGHTMAPS
            // #pragma multi_compile _ LOD_FADE_CROSSFADE
            // #pragma multi_compile_fog
            // #pragma multi_compile_fragment _ DEBUG_DISPLAY
            // #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/ProbeVolumeVariants.hlsl"

            //--------------------------------------
            // GPU Instancing
            // #pragma multi_compile_instancing
            // #pragma instancing_options renderinglayer
            // #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DOTS.hlsl"


            // List all the attributes needed in raytracing shader
            #define ATTRIBUTES_NEED_TEXCOORD0
            #define ATTRIBUTES_NEED_NORMAL
            #define ATTRIBUTES_NEED_TANGENT

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Core.hlsl"


            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/ShaderVariablesRaytracing.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RaytracingIntersection.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RaytracingFragInputs.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RaytracingLighting.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RayTracingCommon.hlsl"


            CBUFFER_START(UnityPerMaterial)
            float3  _BaseColor;
            float4  _BaseMap_ST;
            float   _NormalScale;

            // PBR Properties
            float   _Metallic;
            float   _Smoothness;
            float   _Occlusion;

            // Direct Light
            float4  _SelfLight;
            float   _MainLightColorLerp;
            float   _DirectOcclusion;

            // Shadow
            float4  _ShadowColor;
            float   _ShadowOffset;
            float   _ShadowSmoothNdotL;
            float   _ShadowSmoothScene;
            float   _ShadowStrength;

            // Indirect
            float4  _SelfEnvColor;
            float   _EnvColorLerp;
            float   _IndirDiffUpDirSH;
            float   _IndirDiffIntensity;
            float   _IndirSpecCubeWeight;
            float   _IndirSpecIntensity;

            // Emission
            float4  _EmissionCol;
            // RimLight
            float4  _DirectRimFrontCol;
            float4  _DirectRimBackCol;
            float   _DirectRimWidth;
            float   _PunctualRimWidth;

            // Alpha Test
            float   _Cutoff;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            
            TEXTURE2D(_PBRMask);
            SAMPLER(sampler_PBRMask);



            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RayTracingShaderPassPBRToon.hlsl"

            ENDHLSL
        }

        Pass
        {
            Name "VisibilityDXR"
            Tags{ "LightMode" = "VisibilityDXR" }

            HLSLPROGRAM

            // -------------------------------------
            // Shader Stages
            #pragma only_renderers d3d11 xboxseries ps5
            #pragma raytracing surface_shader

      
            // -------------------------------------
            // Material Keywords
            // #pragma shader_feature_local _NORMALMAP
            // #pragma shader_feature_local _PARALLAXMAP
            // #pragma shader_feature_local _RECEIVE_SHADOWS_OFF
            // #pragma shader_feature_local _ _DETAIL_MULX2 _DETAIL_SCALED
            // #pragma shader_feature_local_fragment _SURFACE_TYPE_TRANSPARENT
            #pragma shader_feature_local _ALPHATEST_ON
            // #pragma shader_feature_local_fragment _ _ALPHAPREMULTIPLY_ON _ALPHAMODULATE_ON
            // #pragma shader_feature_local_fragment _EMISSION
            // #pragma shader_feature_local_fragment _METALLICSPECGLOSSMAP
            // #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
            // #pragma shader_feature_local_fragment _OCCLUSIONMAP
            // #pragma shader_feature_local_fragment _SPECULARHIGHLIGHTS_OFF
            // #pragma shader_feature_local_fragment _ENVIRONMENTREFLECTIONS_OFF
            // #pragma shader_feature_local_fragment _SPECULAR_SETUP

            // -------------------------------------
            // Universal Pipeline keywords
            // #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            // #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            // #pragma multi_compile _ EVALUATE_SH_MIXED EVALUATE_SH_VERTEX
            // #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BLENDING
            // #pragma multi_compile_fragment _ _REFLECTION_PROBE_BOX_PROJECTION
            // #pragma multi_compile_fragment _ _SHADOWS_SOFT _SHADOWS_SOFT_LOW _SHADOWS_SOFT_MEDIUM _SHADOWS_SOFT_HIGH
            // #pragma multi_compile_fragment _ _SCREEN_SPACE_OCCLUSION
            // #pragma multi_compile_fragment _ _DBUFFER_MRT1 _DBUFFER_MRT2 _DBUFFER_MRT3
            // #pragma multi_compile_fragment _ _LIGHT_COOKIES
            // #pragma multi_compile _ _LIGHT_LAYERS
            // #pragma multi_compile _ _FORWARD_PLUS
            // #include_with_pragmas "Packages/com.unity.render-pipelines.core/ShaderLibrary/FoveatedRenderingKeywords.hlsl"
            // #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/RenderingLayers.hlsl"


            // -------------------------------------
            // Unity defined keywords
            // #pragma multi_compile _ LIGHTMAP_SHADOW_MIXING
            // #pragma multi_compile _ SHADOWS_SHADOWMASK
            // #pragma multi_compile _ DIRLIGHTMAP_COMBINED
            // #pragma multi_compile _ LIGHTMAP_ON
            // #pragma multi_compile _ DYNAMICLIGHTMAP_ON
            // #pragma multi_compile _ USE_LEGACY_LIGHTMAPS
            // #pragma multi_compile _ LOD_FADE_CROSSFADE
            // #pragma multi_compile_fog
            // #pragma multi_compile_fragment _ DEBUG_DISPLAY
            // #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/ProbeVolumeVariants.hlsl"

            //--------------------------------------
            // GPU Instancing
            // #pragma multi_compile_instancing
            // #pragma instancing_options renderinglayer
            // #include_with_pragmas "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/DOTS.hlsl"


            // List all the attributes needed in raytracing shader
            #define ATTRIBUTES_NEED_TEXCOORD0
            // #define ATTRIBUTES_NEED_NORMAL
            // #define ATTRIBUTES_NEED_TANGENT

            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/ShaderLibrary/Core.hlsl"


            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/ShaderVariablesRaytracing.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RaytracingIntersection.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RaytracingFragInputs.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RaytracingLighting.hlsl"
            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RayTracingCommon.hlsl"

            float4  _BaseMap_ST;
            float   _Cutoff;
            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            #include "Packages/com.unity.render-pipelines.danbaidong/Shaders/Raytracing/RayTracingShaderPassVisibility.hlsl"

            ENDHLSL
        }
    }

    CustomEditor "UnityEditor.DanbaidongGUI.DanbaidongGUI"
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}