export type Err<Message extends string> = {
  kind: 'Err';
  message: Message;
};

export type Ok = {
  kind: 'Ok';
};

type ValidationResult = Err<string> | Ok;

export type FValidator<T extends ValidationResult> =
  T extends Err<infer Message> ? [error: Message] : [];

export type IsOk<T extends ValidationResult> = T extends Err<infer _> ? 0 : 1;

export type IfOk<T, Otherwise> = T extends Err<infer _> ? T : Otherwise;

export type ValidationLadder<TResults extends ValidationResult[]> =
  TResults extends [
    infer TFirst extends ValidationResult,
    ...infer TRest extends ValidationResult[]
  ]
    ? IfOk<TFirst, ValidationLadder<TRest>>
    : Ok;
