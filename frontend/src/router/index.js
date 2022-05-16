import { createWebHistory, createRouter } from 'vue-router'
import Main from "../views/Main.vue"
import Modify from "../views/Modify.vue"
import Result from "../views/Result.vue"
import Upload from "../views/Upload.vue"

const routes = [
    {
        path: "/",
        name: "Main",
        component: Main
    },
    {
        path: "/upload",
        name: "Upload",
        component: Upload
    },
    {
        path: "/modify",
        name: "Modify",
        component: Modify
    },
    {
        path: "/result",
        name: "Result",
        component: Result
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
});

export default router;