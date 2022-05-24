import { createWebHistory, createRouter } from 'vue-router'
import Main from "../views/Main.vue"
import Modify from "../views/Modify.vue"
import Result from "../views/Result.vue"
import Upload from "../views/Upload.vue"
import Test from "../views/Test.vue"

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
    },
    {
        path: '/test',
        name: "Test",
        component: Test
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
});

export default router;