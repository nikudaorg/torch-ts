import { DType } from './main';

type QuantizedDType = 'qint8' | 'quint8' | 'qint32' | 'quint4x2' | 'quint2x4';

type ShellDType =
  | 'uint1'
  | 'uint2'
  | 'uint3'
  | 'uint4'
  | 'uint5'
  | 'uint6'
  | 'uint7'
  | 'uint16'
  | 'uint32'
  | 'uint64'
  | 'float4_e2m1fn_x2'
  | 'float8_e4m3fn'
  | 'float8_e4m3fnuz'
  | 'float8_e5m2'
  | 'float8_e5m2fnuz'
  | 'float8_e8m0fnu';

type SignedIntegerDType = 'int8' | 'int16' | 'int32' | 'int64';
type IntegralDType = 'uint8' | SignedIntegerDType;
type FloatingDType = 'float16' | 'bfloat16' | 'float32' | 'float64';
type ComplexDType = 'complex32' | 'complex64' | 'complex128';

type UnsupportedPromotionDType = ShellDType | QuantizedDType;

/* -------------------------------------------------------------------------- */
/* Integer promotion                                                          */
/* -------------------------------------------------------------------------- */

type PromoteSignedInteger<
  A extends SignedIntegerDType,
  B extends SignedIntegerDType
> = A extends 'int64'
  ? 'int64'
  : B extends 'int64'
    ? 'int64'
    : A extends 'int32'
      ? 'int32'
      : B extends 'int32'
        ? 'int32'
        : A extends 'int16'
          ? 'int16'
          : B extends 'int16'
            ? 'int16'
            : 'int8';

type PromoteIntegral<
  A extends IntegralDType,
  B extends IntegralDType
> = A extends 'uint8'
  ? B extends 'uint8'
    ? 'uint8'
    : B extends 'int8'
      ? 'int16'
      : B
  : B extends 'uint8'
    ? A extends 'int8'
      ? 'int16'
      : A
    : A extends SignedIntegerDType
      ? B extends SignedIntegerDType
        ? PromoteSignedInteger<A, B>
        : never
      : never;

/* -------------------------------------------------------------------------- */
/* Floating-point promotion                                                   */
/* -------------------------------------------------------------------------- */

type PromoteFloating<
  A extends FloatingDType,
  B extends FloatingDType
> = A extends 'float64'
  ? 'float64'
  : B extends 'float64'
    ? 'float64'
    : A extends 'float32'
      ? 'float32'
      : B extends 'float32'
        ? 'float32'
        : // float16 and bfloat16 together promote to float32.
          A extends B
          ? A
          : 'float32';

/* -------------------------------------------------------------------------- */
/* Complex promotion                                                          */
/* -------------------------------------------------------------------------- */

type ComplexComponent<T extends ComplexDType> = T extends 'complex32'
  ? 'float16'
  : T extends 'complex64'
    ? 'float32'
    : 'float64';

type ComplexFromComponent<T extends FloatingDType> = T extends 'float64'
  ? 'complex128'
  : T extends 'float32'
    ? 'complex64'
    : T extends 'bfloat16'
      ? 'complex64'
      : 'complex32';

type PromoteComplexWithFloating<
  C extends ComplexDType,
  F extends FloatingDType
> = ComplexFromComponent<PromoteFloating<ComplexComponent<C>, F>>;

type PromoteComplex<
  A extends ComplexDType,
  B extends ComplexDType
> = ComplexFromComponent<
  PromoteFloating<ComplexComponent<A>, ComplexComponent<B>>
>;

/* -------------------------------------------------------------------------- */
/* Public type                                                                */
/* -------------------------------------------------------------------------- */

type PromoteSupportedDTypes<
  A extends DType,
  B extends DType
> = A extends UnsupportedPromotionDType
  ? never
  : B extends UnsupportedPromotionDType
    ? never
    : A extends ComplexDType
      ? B extends ComplexDType
        ? PromoteComplex<A, B>
        : B extends FloatingDType
          ? PromoteComplexWithFloating<A, B>
          : A
      : B extends ComplexDType
        ? A extends FloatingDType
          ? PromoteComplexWithFloating<B, A>
          : B
        : A extends FloatingDType
          ? B extends FloatingDType
            ? PromoteFloating<A, B>
            : A
          : B extends FloatingDType
            ? B
            : A extends 'bool'
              ? B
              : B extends 'bool'
                ? A
                : A extends IntegralDType
                  ? B extends IntegralDType
                    ? PromoteIntegral<A, B>
                    : never
                  : never;

/**
 * PyTorch dtype promotion for two dimensioned tensor operands.
 *
 * Returns `never` for shell and quantized dtypes because PyTorch does not
 * define ordinary type promotion for them.
 *
 * The distributive wrappers make unions work:
 * PromoteDType<'int8' | 'float16', 'int16'>
 *   => 'int16' | 'float16'
 */
export type PromoteDType<A extends DType, B extends DType> = A extends unknown
  ? B extends unknown
    ? PromoteSupportedDTypes<A, B>
    : never
  : never;

export type AllowedCastTargets<From extends DType> =
  From extends UnsupportedPromotionDType
    ? never
    : // Boolean can cast to every supported ordinary dtype.
      From extends 'bool'
      ? 'bool' | IntegralDType | FloatingDType | ComplexDType
      : // Integral can cast to any integral, floating, or complex dtype.
        From extends IntegralDType
        ? IntegralDType | FloatingDType | ComplexDType
        : // Floating can cast to any floating or complex dtype.
          From extends FloatingDType
          ? FloatingDType | ComplexDType
          : // Complex can only cast to another complex dtype.
            From extends ComplexDType
            ? ComplexDType
            : never;
