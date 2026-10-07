import "svelte/elements";
declare module "svelte/elements" {
  interface HTMLAttributes<T> {
    "aria-description"?: string;
  }
}
export {};
