import { flushSync, mount } from "svelte";
import PanelEditor from "./components/PanelEditor.svelte";
import PanelShelf from "./components/PanelShelf.svelte";
export class PanelManager {
  editor: ReturnType<typeof PanelEditor>;
  shelf: ReturnType<typeof PanelShelf>;
  constructor(private services: any) {
    this.editor = mount(PanelEditor, {
      target: document.querySelector("#panel-editor-mount")!,
      props: { services },
    });
    this.shelf = mount(PanelShelf, {
      target: document.querySelector("#panel-shelf-items")!,
      props: { services },
    });
    document
      .querySelector("#panel-add")
      ?.addEventListener("click", () => void this.openEditor());
    flushSync();
  }
  openEditor(id: string | null = null) {
    return this.editor.openEditor(id);
  }
  renderShelf() {
    flushSync(() => this.shelf.renderShelf());
  }
  duplicate(id: string) {
    this.services.onDuplicate(id);
  }
  remove(id: string) {
    this.services.onRemove(id);
  }
}
