<template>
  <div class="container">
    <h1>Modify View</h1>
    <div class="row">
      <div class="col-md-7">
        <img
          class="img-fluid rounded mb-4 mb-lg-0"
          :src="circuit_img"
          width="600"
          height="800"
          alt="..."
        />
      </div>
      <div class="col-md-5">
        <div class="row gx-4 gx-lg-5">
          <div class="col">
            <CardBody :title="'저항값 수정하기'">
              <template #footer>
                <div class="row">
                  <button
                    type="button"
                    class="btn btn-primary"
                    data-bs-toggle="modal"
                    data-bs-target="#exampleModal"
                    @click="setResistorValue"
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
                          <div
                            class="row mb-3"
                            v-for="(row, idx) in circuit"
                            :key="`${row}_${idx}`"
                          >
                            <label
                              :for="row['name']"
                              class="col-sm-3 col-form-label"
                              >{{ row["name"] }}</label
                            >
                            <div class="col">
                              <input
                                type="number"
                                class="form-control"
                                :placeholder="row['value']"
                                :id="row['name']"
                                :value="row['value']"
                                @input="setResistorValue($event, row['name'])"
                              />
                            </div>
                          </div>
                        </div>
                        <div class="modal-footer">
                          <button
                            type="button"
                            class="btn btn-secondary"
                            aria-label="Close"
                          >
                            Close
                          </button>
                          <button
                            type="button"
                            class="btn btn-primary"
                            @click="onSaveButton"
                          >
                            Save changes
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </CardBody>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import CardBody from "../components/CardBody.vue";
import ImageModify from "../components/ImageModify.vue";
import axios from "axios";

export default {
  component: { CardBody, ImageModify },
  data() {
    return {
      circuit: null,
      circuit_img: "data:image/png;base64,",
    };
  },
  created() {
    console.log("MOUNTED");
    axios({
      url: "/resistor",
      method: "get",
      headers: {
        "Content-Type": "application/json",
      },
    }).then((response) => {
      const response_data = response.data.data;

      let temp_arr = [];
      for (let i = 0; i < response_data.length; i++) {
        for (let j = 0; j < response_data[i].length; j++) {
          temp_arr.push(response_data[i][j]);
        }
      }
      this.circuit = temp_arr;
    });
  },
  mounted() {
    axios({
      url: "/draw",
      method: "get",
      headers: { "Content-Type": "multipart/form-data" },
    }).then((response) => {
      let img_data = response.data["circuit"];
      this.circuit_img += img_data;
    });
  },
  methods: {
    uploadImg() {
      var image = this.$refs["image"].files[0];
      const url = URL.createObjectURL(image);
      this.image = url;
    },
    setResistorValue(event, name) {
      this.circuit.map((row) => {
        if (row["name"] === name) {
          row["value"] = parseInt(event.target.value);
        }
      });
    },
    onSaveButton() {
      axios({
        method: "post",
        url: "/resistor",
        headers: {
          "Content-Type": "application/json",
        },
        data: JSON.stringify(this.circuit),
      }).then(() => {
        this.$router.go(this.$router.currentRoute);
      });
    },
  },
  components: { CardBody },
};
</script>
