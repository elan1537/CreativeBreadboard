<template>
  <div class="row gx-4 gx-lg-5 align-items-center my-5">
    <div class="col-lg-5">
      <div id="root">
        <h2 class="title">Insert image</h2>
        <hr />
        <div class="contents">
          <div class="upload-box">
            <div id="drop-file" class="drag-file">
              <img class="img-fluid rounded mb-4 mb-lg-0" src="" alt="..." />
            </div>
            <img src=" " alt=" 회로도 " class="preview" />
          </div>
          <label class="file-label" for="chooseFile">Choose File</label>
          <input
            id="chooseFile"
            class="file"
            type="file"
            onchange="dropFile.handleFiles(this.files)"
            accept="image/png, image/jpeg, image/gif"
          />
        </div>
      </div>
    </div>
    <button onclick="location.href='paris baguette - 2페이지.html'">
      To 2PAGE
    </button>
    <!-- Modal -->
    <div
      id="staticBackdrop"
      class="modal fade"
      data-bs-backdrop="static"
      data-bs-keyboard="false"
      tabindex="-1"
      aria-labelledby="staticBackdropLabel"
      aria-hidden="true"
    >
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 id="staticBackdropLabel" class="modal-title">Modal title</h5>
            <button
              type="button"
              class="btn-close"
              data-bs-dismiss="modal"
              aria-label="Close"
            ></button>
          </div>
          <div class="modal-body">...</div>
          <div class="modal-footer">
            <button
              type="button"
              class="btn btn-secondary"
              data-bs-dismiss="modal"
            >
              Close
            </button>
            <button type="button" class="btn btn-primary">Understood</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  name: "ComponentImageViewer",
};
</script>
