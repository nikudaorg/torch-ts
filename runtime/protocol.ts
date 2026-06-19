import type { Socket } from 'node:net';

export const PROTOCOL_VERSION = 1;

export type JsonPrimitive = null | boolean | number | string;
export type JsonValue =
  | JsonPrimitive
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface TensorReference {
  [key: string]: JsonValue;
  $tensor: number;
}

export interface TensorResult {
  [key: string]: JsonValue;
  kind: 'tensor';
  handle: number;
  shape: number[];
  dtype: string;
  device: string;
  repr: string;
}

export interface RpcRequest {
  id: number;
  op: string;
  args?: JsonValue[];
  kwargs?: Record<string, JsonValue>;
}

export type RpcResponse =
  | { id: number; ok: true; result: JsonValue | TensorResult }
  | {
      id: number;
      ok: false;
      error: { type: string; message: string; traceback?: string };
    };

export function encodeFrame(value: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(value), 'utf8');
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32BE(body.length);
  return Buffer.concat([header, body]);
}

export class FrameDecoder {
  private buffered = Buffer.alloc(0);

  push(chunk: Buffer): unknown[] {
    this.buffered = Buffer.concat([this.buffered, chunk]);
    const messages: unknown[] = [];

    while (this.buffered.length >= 4) {
      const size = this.buffered.readUInt32BE(0);
      if (this.buffered.length < size + 4) break;

      const body = this.buffered.subarray(4, size + 4);
      this.buffered = this.buffered.subarray(size + 4);
      messages.push(JSON.parse(body.toString('utf8')));
    }

    return messages;
  }
}

export function writeFrame(socket: Socket, value: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.write(encodeFrame(value), error => {
      if (error) reject(error);
      else resolve();
    });
  });
}
