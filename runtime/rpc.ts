import { spawn, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { mkdir, rm } from 'node:fs/promises';
import { createConnection, type Socket } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  FrameDecoder,
  PROTOCOL_VERSION,
  type JsonValue,
  type RpcRequest,
  type RpcResponse,
  writeFrame
} from './protocol.ts';

export interface RpcClientOptions {
  pythonPath: string;
  workerPath: string;
  endpoint?: string;
  startupTimeoutMs: number;
  env?: NodeJS.ProcessEnv;
}

interface PendingCall {
  resolve(value: JsonValue): void;
  reject(error: Error): void;
}

export class PythonRpcClient {
  private readonly options: RpcClientOptions;
  private readonly endpoint: string;
  private readonly socketDirectory: string | undefined;
  private readonly pending = new Map<number, PendingCall>();
  private readonly ready: Promise<void>;
  private child?: ChildProcess;
  private socket?: Socket;
  private nextId = 1;
  private closed = false;
  private stderr = '';

  constructor(options: RpcClientOptions) {
    this.options = options;
    const generated = createEndpoint();
    this.endpoint = options.endpoint ?? generated.endpoint;
    this.socketDirectory =
      options.endpoint === undefined ? generated.directory : undefined;
    this.ready = this.start();
  }

  async call(
    op: string,
    args: JsonValue[] = [],
    kwargs: Record<string, JsonValue> = {}
  ): Promise<JsonValue> {
    await this.ready;
    if (this.closed || !this.socket) {
      throw new Error('The torch Python runtime is closed');
    }

    const id = this.nextId++;
    const request: RpcRequest = { id, op, args, kwargs };
    const result = new Promise<JsonValue>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });

    try {
      await writeFrame(this.socket, request);
    } catch (error) {
      this.pending.delete(id);
      throw error;
    }

    return result;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;

    try {
      await this.ready;
      if (this.socket && !this.socket.destroyed) {
        const id = this.nextId++;
        await writeFrame(this.socket, { id, op: '__close__' });
      }
    } catch {
      // Closing a worker that failed to start is still safe.
    }

    this.socket?.destroy();
    this.child?.kill();
    await this.cleanupEndpoint();
    this.rejectAll(new Error('The torch Python runtime was closed'));
  }

  private async start(): Promise<void> {
    if (this.socketDirectory) {
      await mkdir(this.socketDirectory, { recursive: true });
    }

    const child = spawn(
      this.options.pythonPath,
      [
        this.options.workerPath,
        '--endpoint',
        this.endpoint,
        '--protocol-version',
        String(PROTOCOL_VERSION)
      ],
      {
        stdio: ['ignore', 'ignore', 'pipe'],
        env: { ...process.env, ...this.options.env }
      }
    );
    this.child = child;
    child.stderr?.setEncoding('utf8');
    child.stderr?.on('data', chunk => {
      this.stderr = (this.stderr + String(chunk)).slice(-16_384);
    });

    const earlyExit = new Promise<never>((_, reject) => {
      child.once('error', reject);
      child.once('exit', (code, signal) => {
        reject(
          new Error(
            `Torch worker exited before connecting (code=${code}, signal=${signal})` +
              (this.stderr ? `\n${this.stderr}` : '')
          )
        );
      });
    });

    try {
      const socket = await Promise.race([
        connectWithRetry(this.endpoint, this.options.startupTimeoutMs),
        earlyExit
      ]);
      this.socket = socket;
      this.attachSocket(socket);
    } catch (error) {
      child.kill();
      await this.cleanupEndpoint();
      throw error;
    }
  }

  private attachSocket(socket: Socket): void {
    const decoder = new FrameDecoder();
    socket.on('data', chunk => {
      try {
        const buffer = typeof chunk === 'string' ? Buffer.from(chunk) : chunk;
        for (const value of decoder.push(buffer)) {
          this.handleResponse(value as RpcResponse);
        }
      } catch (error) {
        socket.destroy();
        this.rejectAll(
          error instanceof Error ? error : new Error(String(error))
        );
      }
    });
    socket.on('error', error => this.rejectAll(error));
    socket.on('close', () => {
      if (!this.closed) {
        this.rejectAll(
          new Error(
            'Connection to the torch Python runtime closed unexpectedly' +
              (this.stderr ? `\n${this.stderr}` : '')
          )
        );
      }
    });
  }

  private handleResponse(response: RpcResponse): void {
    const pending = this.pending.get(response.id);
    if (!pending) return;
    this.pending.delete(response.id);

    if (response.ok) {
      pending.resolve(response.result);
      return;
    }

    const error = new Error(
      `${response.error.type}: ${response.error.message}` +
        (response.error.traceback ? `\n${response.error.traceback}` : '')
    );
    error.name = response.error.type;
    pending.reject(error);
  }

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  private async cleanupEndpoint(): Promise<void> {
    if (this.socketDirectory) {
      await rm(this.socketDirectory, { recursive: true, force: true });
    }
  }
}

function createEndpoint(): { endpoint: string; directory?: string } {
  const nonce = randomBytes(8).toString('hex');
  if (process.platform === 'win32') {
    return { endpoint: `\\\\.\\pipe\\torch-ts-${process.pid}-${nonce}` };
  }

  const directory = join(tmpdir(), `torch-ts-${process.pid}-${nonce}`);
  return { endpoint: join(directory, 'worker.sock'), directory };
}

function connectWithRetry(endpoint: string, timeoutMs: number): Promise<Socket> {
  const startedAt = Date.now();

  return new Promise((resolve, reject) => {
    const attempt = (): void => {
      const socket = createConnection(endpoint);
      const onError = (error: NodeJS.ErrnoException): void => {
        socket.destroy();
        if (
          Date.now() - startedAt < timeoutMs &&
          ['ENOENT', 'ECONNREFUSED', 'EPIPE'].includes(error.code ?? '')
        ) {
          setTimeout(attempt, 20);
        } else {
          reject(error);
        }
      };
      socket.once('error', onError);
      socket.once('connect', () => {
        socket.off('error', onError);
        resolve(socket);
      });
    };

    attempt();
  });
}
