<template>
  <div class="container px-4 px-lg-5">
    <h1>Modify</h1>
    <div>
      <div class="row gx-4 gx-lg-5 align-items-center my-5">
        <!--src에 저장되는 사진의 이름이 뜬다 like result.png나 jpg-->
        <div class="col-lg-7">
          <img
            class="img-fluid rounded mb-4 mb-lg-0"
            src="result.jpg"
            alt="..."
          />
        </div>
        <div class="col-lg-5">
          <div class="row gx-4 gx-lg-5">
            <div class="col-md-6 mb-7">
              <div class="card h-500">
                <div class="card-body">
                  <h4><strong>이미지를 변경</strong></h4>
                  <p class="card-text">가나다가나다가나다</p>
                </div>
                <div class="card-footer">
                  <div class="row">
                    <div class="col">
                      <a
                        class="btn btn-secondary btn-sm"
                        onclick="location.href='paris baguette - 1페이지.html'"
                        >Modify</a
                      >
                    </div>
                    <div class="col">
                      <a class="btn btn-secondary btn-sm" href="#!">Complete</a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div class="col-md-6 mb-7">
              <div class="card h-500">
                <div class="card-body">
                  <h4><strong>저항 값 수정</strong></h4>
                  <p class="card-text">가나다</p>
                </div>
                <div class="card-footer">
                  <div class="row">
                    <div class="col">
                      <!-- Button trigger modal -->
                      <button
                        type="button"
                        class="btn btn-secondary"
                        data-bs-toggle="modal"
                        data-bs-target="#exampleModal"
                      >
                        modify
                      </button>
                      <!-- Modal -->
                      <div
                        class="modal fade"
                        id="exampleModal"
                        tabindex="-1"
                        aria-labelledby="exampleModalLabel"
                        aria-hidden="true"
                      >
                        <div class="modal-dialog">
                          <div class="modal-content">
                            <div class="modal-header">
                              <h2 class="modal-title" id="exampleModalLabel">
                                저항값을 입력하세요
                              </h2>
                              <button
                                type="button"
                                class="btn-close"
                                data-bs-dismiss="modal"
                                aria-label="Close"
                              ></button>
                            </div>
                            <div class="modal-body">
                              <table border="1px" id="data_table">
                                <tbody>
                                  <tr>
                                    <th>R11</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                    <th>R12</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                    <th>R13</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                  </tr>
                                  <tr>
                                    <th>R21</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                    <th>R22</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                    <th>R23</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                  </tr>
                                  <tr>
                                    <th>R31</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                    <th>R32</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                    <th>R33</th>
                                    <td>
                                      <input
                                        id="h1"
                                        type="text"
                                        name="h1"
                                        value="0"
                                        style="width: 70px"
                                        onchange="calc();"
                                      />
                                    </td>
                                  </tr>
                                </tbody>
                              </table>
                            </div>
                            <div class="modal-footer">
                              <button type="button" class="btn btn-secondary">
                                Save changes
                              </button>
                              <button
                                type="button"
                                class="btn btn-secondary"
                                data-bs-dismiss="modal"
                              >
                                Close
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div class="col">
                      <a class="btn btn-secondary btn-sm" href="#!">Complete</a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  components: {},
  data() {
    return {
      image: "",
    };
  },
  methods: {
    uploadImg() {
      var image = this.$refs["image"].files[0];

      const url = URL.createObjectURL(image);
      this.image = url;
    },
  },
};
</script>
